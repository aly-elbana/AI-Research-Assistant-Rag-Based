import argparse
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

try:
    from rank_bm25 import BM25Okapi
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    BM25Okapi = None

from langchain_chroma import Chroma
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DEFAULT_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "deepseek-r1:7b-qwen-distill-q4_k_m")
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "OLLAMA_EMBEDDING_MODEL",
    "nomic-embed-text:latest",
)
DEFAULT_COLLECTION_NAME = os.getenv("RESEARCH_COLLECTION_NAME", "research_doc")


class ResearchResponse(BaseModel):
    """Structured research answer returned to the caller."""

    answer: str = Field(description="The answer to the user's research question.")
    confidence: str = Field(
        description="The confidence level of the answer: high, medium, or low."
    )
    confidence_percentage: float = Field(
        description=(
            "The system's estimated confidence in the answer, expressed as a "
            "percentage between 0 and 100."
        ),
        ge=0,
        le=100,
    )
    sources: List[str] = Field(
        description="A list of source documents used to generate the answer."
    )
    key_quotes: List[str] = Field(
        default_factory=list,
        description="Relevant quotations extracted from the source documents.",
    )
    follow_up_questions: List[str] = Field(
        default_factory=list,
        description="Suggested follow-up questions for further research.",
    )


class AIResearchAssistant:
    """Manage document ingestion, vector storage, retrieval, and conversations."""

    def __init__(
        self,
        persist_dir: str = "./research_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        collection_name: Optional[str] = None,
        reset_on_init: bool = False,
    ) -> None:
        """Initialize the embedding model, text splitter, vector store, and chat history."""

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be non-negative and smaller than chunk_size."
            )

        self.persist_dir = persist_dir
        self.llm_model = llm_model or DEFAULT_LLM_MODEL
        self.embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL
        self.collection_name = collection_name or DEFAULT_COLLECTION_NAME

        self.llm = ChatOllama(model=self.llm_model)
        self.embeddings_model = OllamaEmbeddings(model=self.embedding_model)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.vector_store = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings_model,
            collection_name=self.collection_name,
        )

        self.documents: List[Document] = []
        self.retriever = RunnableLambda(lambda query: [])
        self.session_store: Dict[str, InMemoryChatMessageHistory] = {}

        if reset_on_init:
            self.reset_vector_store()
        else:
            self.documents = self._load_indexed_documents()
            self.retriever = self._build_retriever()

    def _load_indexed_documents(self) -> List[Document]:
        """Load all indexed document chunks from Chroma."""

        results = self.vector_store.get(include=["documents", "metadatas"])
        contents = results.get("documents") or []
        metadatas = results.get("metadatas") or []

        documents: List[Document] = []
        for content, metadata in zip(contents, metadatas):
            if content is not None:
                documents.append(
                    Document(
                        page_content=content,
                        metadata=metadata or {},
                    )
                )
        return documents

    def reset_vector_store(self) -> None:
        """Delete the current collection and recreate an empty one."""

        try:
            self.vector_store.reset_collection()
        except Exception as exc:  # pragma: no cover - depends on Chroma backend state
            print(f"Warning: could not reset collection cleanly: {exc}")

        self.documents = []
        self.retriever = self._build_retriever()
        print("Vector store reset successfully.")

    def add_text(
        self,
        text: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add a single text snippet to the vector store."""

        document = Document(
            page_content=text,
            metadata={"source": source, **(metadata or {})},
        )
        return self.add_documents([document])

    def add_texts(
        self,
        texts: List[str],
        source: str,
    ) -> int:
        """Add multiple text snippets from a common source."""

        documents = [
            Document(page_content=text, metadata={"source": source})
            for text in texts
        ]
        return self.add_documents(documents)

    def add_documents(
        self,
        documents: List[Document],
        source_name: Optional[str] = None,
    ) -> int:
        """Split documents into chunks and add them to Chroma."""

        if not documents:
            return 0

        normalized_documents = [
            Document(
                page_content=document.page_content,
                metadata={**(document.metadata or {}), "source": source_name or document.metadata.get("source", "unknown")},
            )
            for document in documents
        ]

        chunks = self.splitter.split_documents(normalized_documents)

        indexed_at = datetime.now().isoformat()
        for index, chunk in enumerate(chunks):
            chunk.metadata["indexed_at"] = indexed_at
            chunk.metadata["chunk_index"] = index

        if chunks:
            self.vector_store.add_documents(chunks)
            self.documents = self._load_indexed_documents()
            self.retriever = self._build_retriever()
            print(f"Added {len(chunks)} chunks from {len(documents)} documents.")

        return len(chunks)

    def get_document_count(self) -> int:
        """Return total number of stored chunks."""

        return self.vector_store._collection.count()

    def list_sources(self) -> List[str]:
        """Return sorted, unique source names from stored metadata."""

        results = self.vector_store.get()
        metadatas = results.get("metadatas") or []

        sources = set()
        for metadata in metadatas:
            source = metadata.get("source") if isinstance(metadata, dict) else None
            if isinstance(source, str) and source:
                sources.add(source)
        return sorted(sources)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text for BM25 retrieval."""

        if not text:
            return []
        return re.findall(r"\w+", text.lower())

    def _build_retriever(self, basic: bool = False):
        """Build a hybrid BM25 and vector retriever."""

        vector_retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 10,
                "fetch_k": 30,
                "lambda_mult": 0.5,
            },
        )

        if basic or BM25Okapi is None:
            return vector_retriever

        if not self.documents:
            return RunnableLambda(lambda query: [])

        tokenized_documents = [
            self._tokenize(document.page_content) for document in self.documents
        ]
        bm25 = BM25Okapi(tokenized_documents)

        def retrieve(query: str) -> List[Document]:
            """Combine BM25 and vector retrieval using reciprocal rank fusion."""

            query_tokens = self._tokenize(query)
            bm25_scores = bm25.get_scores(query_tokens) if query_tokens else [0.0] * len(self.documents)

            bm25_indices = sorted(
                range(len(bm25_scores)),
                key=lambda index: bm25_scores[index],
                reverse=True,
            )[:10]
            bm25_results = [self.documents[index] for index in bm25_indices]

            vector_results: List[Document] = []
            try:
                vector_results = vector_retriever.invoke(query)
            except Exception:
                vector_results = []

            rankings: Dict[tuple, Dict[str, Any]] = {}
            for weight, results in [(0.4, bm25_results), (0.6, vector_results)]:
                for rank, document in enumerate(results, start=1):
                    key = (
                        document.metadata.get("source"),
                        document.page_content,
                    )
                    if key not in rankings:
                        rankings[key] = {"document": document, "score": 0.0}
                    rankings[key]["score"] += weight / (60 + rank)

            ranked_documents = sorted(
                rankings.values(),
                key=lambda item: item["score"],
                reverse=True,
            )

            return [item["document"] for item in ranked_documents[:4]]

        return RunnableLambda(retrieve)

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Retrieve an existing session or create a new one."""

        if session_id not in self.session_store:
            self.session_store[session_id] = InMemoryChatMessageHistory()
        return self.session_store[session_id]

    def clear_session(self, session_id: str) -> None:
        """Clear conversation history for a specific session."""

        if session_id in self.session_store:
            self.session_store[session_id].clear()
            print(f"Cleared Session: {session_id}")

    def get_session_history_display(self, session_id: str) -> List[Dict[str, Any]]:
        """Return formatted conversation history for display."""

        if session_id not in self.session_store:
            return []

        return [
            {
                "role": "human" if isinstance(message, HumanMessage) else "assistant",
                "content": message.content,
            }
            for message in self.session_store[session_id].messages
        ]

    def search(self, query: str) -> List[Document]:
        """Retrieve relevant document chunks for a query."""

        if not query or not query.strip():
            return []

        try:
            return self.retriever.invoke(query)
        except Exception:
            return []

    def _fallback_answer(self, question: str, docs: List[Document], error: Optional[Exception] = None) -> str:
        """Return a safe, grounded fallback when the LLM cannot generate a response."""

        if not docs:
            return (
                "I don't have enough information in the indexed documents to answer "
                f"this question: '{question}'."
            )

        source_names = sorted(
            {
                document.metadata.get("source", "unknown")
                for document in docs
                if isinstance(document.metadata.get("source", "unknown"), str)
            }
        )
        if error:
            return (
                "The model could not generate a final answer, but the document context "
                f"suggests the answer is grounded in {source_names or 'the available sources'}."
            )
        return (
            "I found relevant context in the indexed sources, but the model response "
            f"was not available. Sources: {source_names or 'unknown'}."
        )

    def ask(
        self,
        question: str,
        session_id: str = "session_id",
    ) -> str:
        """Retrieve relevant documents and generate a contextual answer."""

        history = self._get_session_history(session_id)
        docs = self.search(question)
        context = format_search_results(docs)

        if not question or not question.strip():
            return "Please provide a research question."

        if not docs:
            response = self._fallback_answer(question, docs)
            history.add_messages([
                HumanMessage(content=question),
                AIMessage(content=response),
            ])
            return response

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an AI research assistant. Answer the user's question using "
                    "the provided context. All answers must be supported by the source "
                    "material. If the context doesn't contain the answer, say so clearly.\n\n"
                    "Context:\n{context}",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        try:
            response = (prompt | self.llm | StrOutputParser()).invoke(
                {
                    "context": context,
                    "question": question,
                    "history": history.messages[-10:],
                }
            )
        except Exception as exc:
            response = self._fallback_answer(question, docs, exc)

        history.add_messages([
            HumanMessage(content=question),
            AIMessage(content=response),
        ])
        return response

    def research(self, question: str, session_id: str = "session_id") -> ResearchResponse:
        """Return a validated structured research response."""

        docs = self.search(question)
        answer = self.ask(question=question, session_id=session_id)

        sources = sorted(
            {
                doc.metadata.get("source", "unknown")
                for doc in docs
                if isinstance(doc.metadata.get("source", "unknown"), str)
            }
        )

        key_quotes = extract_key_quotes(docs, question)
        if "don't have enough information" in answer.lower():
            confidence = "low"
            percentage = 25.0
        elif len(docs) >= 2:
            confidence = "high"
            percentage = min(95.0, 55.0 + (len(docs) * 10))
        else:
            confidence = "medium"
            percentage = 60.0

        follow_up_questions = [
            f"Can you expand on the {question.strip()[:40]} details?",
            "What other sources or examples should I review?",
        ]

        return ResearchResponse(
            answer=answer,
            confidence=confidence,
            confidence_percentage=percentage,
            sources=sources,
            key_quotes=key_quotes,
            follow_up_questions=follow_up_questions,
        )


def extract_key_quotes(documents: List[Document], question: str) -> List[str]:
    """Extract short, relevant quotations from the top matching documents."""

    if not documents:
        return []

    question_tokens = set(re.findall(r"\w+", question.lower()))
    quotes: List[str] = []

    for document in documents:
        sentences = re.split(r"(?<=[.!?])\s+", document.page_content)
        for sentence in sentences:
            sentence_tokens = set(re.findall(r"\w+", sentence.lower()))
            if not sentence_tokens:
                continue
            overlap = sentence_tokens & question_tokens
            if overlap and len(sentence) <= 260:
                quotes.append(sentence.strip())
                if len(quotes) >= 3:
                    return quotes
    return quotes[:3]


def load_sample_documents() -> List[Document]:
    """Create sample documents about RAG, LLMs, and MCP."""

    return [
        Document(
            page_content=(
                "Retrieval-Augmented Generation (RAG) is an architecture that combines "
                "information retrieval with language model generation. Instead of relying "
                "only on knowledge learned during training, a RAG application retrieves "
                "relevant documents from an external knowledge source and provides their "
                "contents to the language model as context. A typical RAG pipeline includes "
                "document loading, text splitting, embedding generation, vector storage, "
                "retrieval, prompt construction, and answer generation. Vector databases "
                "such as Chroma can store document embeddings and retrieve semantically "
                "similar chunks. RAG can improve grounding in private or specialized data, "
                "but its answer quality depends on document quality, retrieval relevance, "
                "chunking strategy, and prompt design."
            ),
            metadata={"source": "rag_guide.txt", "topic": "RAG"},
        ),
        Document(
            page_content=(
                "A Large Language Model (LLM) is a neural network trained on large "
                "collections of text to process and generate language. LLMs can answer "
                "questions, summarize documents, write code, and extract structured "
                "information. Models may run remotely through a hosted service or locally "
                "using tools such as Ollama. Local inference can offer more control over "
                "data and infrastructure, but performance depends on available memory, "
                "model size, quantization, and hardware. An LLM does not automatically know "
                "the contents of a private document collection. A RAG system can retrieve "
                "relevant documents and include them in the model's prompt. Model outputs "
                "should still be evaluated for factual accuracy, relevance, and consistency."
            ),
            metadata={"source": "llm_guide.txt", "topic": "LLM"},
        ),
        Document(
            page_content=(
                "The Model Context Protocol (MCP) is a protocol for connecting AI "
                "applications to external tools and data sources through a standardized "
                "interface. An MCP server can expose tools, resources, and prompts to "
                "compatible clients. For example, a server might provide a tool that "
                "searches a database, retrieves project documents, or queries a business "
                "system. An AI agent can decide when to call an available tool, inspect its "
                "result, and use that result in a larger workflow. MCP does not itself "
                "guarantee that an application is an autonomous agent; the client or agent "
                "runtime controls planning and tool execution. A production MCP integration "
                "should consider permissions, input validation, error handling, logging, "
                "and confirmation before sensitive actions."
            ),
            metadata={"source": "mcp_agents_guide.txt", "topic": "MCP and AI Agents"},
        ),
    ]


def print_section(title: str) -> None:
    """Print a formatted section header."""

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_vector_store_summary(assistant: AIResearchAssistant, total_chunks: int) -> None:
    """Display vector store statistics."""

    print_section("VECTOR STORE SUMMARY")
    print(f"Total chunks: {assistant.get_document_count()}")
    print(f"Chunks added in this run: {total_chunks}")
    print(f"Sources: {assistant.list_sources()}")


def format_search_results(results: List[Document]) -> str:
    """Format retrieved documents into a context string."""

    if not results:
        return "No relevant documents were retrieved."

    formatted_results: List[str] = []
    for index, document in enumerate(results, start=1):
        formatted_results.append(
            f"Result {index}\n"
            f"Source: {document.metadata.get('source', 'unknown')}\n"
            f"Topic: {document.metadata.get('topic', 'unknown')}\n"
            f"Chunk: {document.metadata.get('chunk_index', 'unknown')}\n"
            f"Content:\n{document.page_content}"
        )
    return "\n\n".join(formatted_results)


def print_search_results(query: str, results: List[Document]) -> None:
    """Display retrieved documents in the console."""

    print_section(f"QUERY: {query}")
    if not results:
        print("No relevant documents were retrieved.")
        return
    print(format_search_results(results))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments with safe defaults for a demo or production use."""

    parser = argparse.ArgumentParser(description="RAG research assistant demo")
    parser.add_argument("--persist-dir", default="./research_db", help="Directory for the Chroma vector store.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for splitting documents.")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap for splitting documents.")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="Ollama LLM model to use.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="Ollama embedding model to use.")
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME, help="Chroma collection name.")
    parser.add_argument("--reset-db", action="store_true", help="Reset the vector store before indexing demo documents.")
    parser.add_argument("--query", default=None, help="Run a single research query after indexing.")
    return parser.parse_args()


def main() -> None:
    """Run the research assistant, optionally indexing sample documents and answering queries."""

    args = parse_args()

    assistant = AIResearchAssistant(
        persist_dir=args.persist_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        collection_name=args.collection_name,
        reset_on_init=args.reset_db,
    )

    if args.reset_db:
        assistant.reset_vector_store()

    documents = load_sample_documents()
    if not assistant.get_document_count():
        total_chunks = assistant.add_documents(documents)
        print_vector_store_summary(assistant, total_chunks)
    else:
        print_vector_store_summary(assistant, 0)

    session_id = "research_session_1"

    queries = [
        "How does Retrieval-Augmented Generation work?",
        "What are the advantages and limitations of running an LLM locally?",
        "How can an AI application use MCP to connect to external tools?",
    ]

    if args.query:
        queries = [args.query]

    for query in queries:
        print_section(f"QUERY: {query}")
        results = assistant.search(query)
        print("\nRETRIEVED DOCUMENTS:")
        print_search_results(query, results)
        answer = assistant.ask(question=query, session_id=session_id)
        print("\nGENERATED ANSWER:")
        print(answer)

    if queries:
        print_section("CONVERSATION HISTORY")
        for message in assistant.get_session_history_display(session_id):
            print(f"\n{message['role'].upper()}:")
            print(message["content"])


if __name__ == "__main__":
    main()