# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma


VECTOR_STORE_DIR = "chroma_vector_store"


def load_and_preprocess_pdf(pdf_path: str):
    """Load a PDF and split it into smaller text chunks."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_documents(documents)

    # Add useful metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks


def create_or_load_vector_store(chunks=None):
    """Create a new Chroma vector store or load an existing one from disk."""
    embeddings = OpenAIEmbeddings()
    persist_directory = VECTOR_STORE_DIR

    if os.path.exists(persist_directory):
        print("Loading existing vector store...")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
    else:
        print("Creating new vector store...")
        if chunks is None:
            raise ValueError("Chunks must be provided to create a new vector store.")
        vectorstore = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=persist_directory,
        )
    return vectorstore


def create_rag_components(vectorstore):
    """Return retriever and LLM (no chain objects)."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(
        model="gpt-4o-mini",  # or "gpt-4o"
        temperature=0,
    )

    return retriever, llm


def query_rag(retriever, llm, query: str):
    """Manually run retrieval + generation without using langchain.chains."""
    # 1) Retrieve relevant documents
    docs = retriever.get_relevant_documents(query)

    # 2) Build context string from retrieved docs
    context = "\n\n".join(doc.page_content for doc in docs)

    # 3) Create a prompt for the LLM
    prompt = f"""
You are a helpful assistant. Answer the question based ONLY on the context below.
If the answer is not in the context, say "I don't know" and do not guess.

<context>
{context}
</context>

Question: {query}
Answer:
"""

    # 4) Call the LLM
    response = llm.invoke(prompt)

    print("\nAnswer:")
    print(response.content)

    print("\nSource documents:")
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "Unknown page")
        chunk_index = doc.metadata.get("chunk_index", "N/A")
        print(f"- [{i}] Source: {source}, Page: {page}, Chunk: {chunk_index}")


def main():
    pdf_path = "yourdocument.pdf"  # change this to your PDF filename

    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return

    print("Loading and processing PDF...")
    chunks = load_and_preprocess_pdf(pdf_path)

    print("Creating/loading vector store...")
    vectorstore = create_or_load_vector_store(chunks)

    print("Setting up RAG components (retriever + LLM)...")
    retriever, llm = create_rag_components(vectorstore)

    print("RAG chatbot ready. Ask questions about the PDF!")

    while True:
        query = input("\nAsk a question (or type 'exit' to quit): ")
        if query.lower().strip() == "exit":
            break
        try:
            query_rag(retriever, llm, query)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set. Please set it in your environment.")
    else:
        main()
