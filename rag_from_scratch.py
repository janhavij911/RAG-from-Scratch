"""
rag_from_scratch.py
A Retrieval-Augmented Generation system built without any RAG framework
(no LangChain, no managed vector database). Every piece is implemented by
hand: chunking, embedding, cosine similarity search, and grounded generation --
using models deployed in Azure AI Foundry.

Knowledge base: the same 15 short articles and chunking strategy (60 words,
15-word overlap) used in the Databricks Bronze-Silver-Gold pipeline project,
so this RAG system is a natural continuation of that work.

Setup:
    pip install azure-ai-inference python-dotenv numpy

Environment variables required (.env file):
    AZURE_FOUNDRY_ENDPOINT=https://<your-resource>.services.ai.azure.com/models
    AZURE_FOUNDRY_KEY=<your-key>
    AZURE_FOUNDRY_CHAT_DEPLOYMENT=gpt-5.6-sol
    AZURE_FOUNDRY_EMBEDDING_DEPLOYMENT=text-embedding-3-small
"""

import os
import json
import numpy as np
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient, EmbeddingsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

load_dotenv()

ENDPOINT = os.getenv("AZURE_FOUNDRY_ENDPOINT")
KEY = os.getenv("AZURE_FOUNDRY_KEY")
CHAT_DEPLOYMENT = os.getenv("AZURE_FOUNDRY_CHAT_DEPLOYMENT", "gpt-5.6-sol")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_FOUNDRY_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

chat_client = ChatCompletionsClient(endpoint=ENDPOINT, credential=AzureKeyCredential(KEY))
embed_client = EmbeddingsClient(endpoint=ENDPOINT, credential=AzureKeyCredential(KEY))

# ---------------------------------------------------------------------------
# Step 1: Knowledge base -- same articles and chunking logic as the Databricks
# Gold layer project, reused here as the RAG system's source documents.
# ---------------------------------------------------------------------------

ARTICLES = [
    {"doc_id": 1, "title": "Introduction to Cloud Computing", "text": "Cloud computing delivers computing services over the internet, including servers, storage, databases, networking, software, and analytics. Instead of owning physical infrastructure, organizations rent access to these resources from a cloud provider. This model offers flexibility, since businesses can scale resources up or down based on demand. The three primary service models are Infrastructure as a Service, Platform as a Service, and Software as a Service."},
    {"doc_id": 2, "title": "What is Data Engineering", "text": "Data engineering is the practice of designing and building systems that collect, store, and analyze data at scale. Data engineers build pipelines that transform raw data into a usable format for analysts and data scientists. Core responsibilities include building ETL and ELT pipelines, managing data warehouses and lakes, and ensuring data quality and reliability."},
    {"doc_id": 3, "title": "Understanding Large Language Models", "text": "Large language models, or LLMs, are AI systems trained on massive amounts of text data to understand and generate human-like language. They work by predicting the next word in a sequence based on patterns learned during training. Modern LLMs like Claude and GPT can perform a wide range of tasks including summarization, translation, question answering, and code generation."},
    {"doc_id": 4, "title": "The Basics of Delta Lake", "text": "Delta Lake is an open-source storage layer that brings reliability to data lakes. It adds ACID transactions, scalable metadata handling, and unifies streaming and batch data processing on top of existing cloud storage. One of its most useful features is time travel, which allows users to query previous versions of a table."},
    {"doc_id": 5, "title": "What is Retrieval-Augmented Generation", "text": "Retrieval-Augmented Generation, or RAG, is a technique that combines a retrieval system with a language model to produce more accurate and grounded responses. Instead of relying solely on what the model learned during training, RAG retrieves relevant documents or chunks from an external knowledge base and provides them as context to the model before generating an answer. This approach helps reduce hallucinations."},
    {"doc_id": 6, "title": "Introduction to Vector Embeddings", "text": "Vector embeddings are numerical representations of text, images, or other data that capture semantic meaning in a high-dimensional space. Similar pieces of content end up close together in this space, which allows systems to find related information through similarity search rather than exact keyword matching. Embeddings are a core building block of modern search and recommendation systems."},
    {"doc_id": 7, "title": "Azure AI Document Intelligence Overview", "text": "Azure AI Document Intelligence is a cloud service that extracts structured data from documents such as invoices, receipts, and forms. It uses prebuilt and custom machine learning models to identify fields like vendor names, dates, totals, and line items from scanned or digital documents."},
    {"doc_id": 8, "title": "The Medallion Architecture Pattern", "text": "The medallion architecture is a data design pattern used to organize data in a lakehouse into three layers: bronze, silver, and gold. The bronze layer stores raw, unprocessed data exactly as it was ingested. The silver layer contains cleaned and validated data with duplicates removed. The gold layer holds business-ready, aggregated data optimized for reporting or downstream applications."},
    {"doc_id": 9, "title": "Why Data Quality Checks Matter", "text": "Data quality checks are automated validations that catch problems in a dataset before they reach downstream users or systems. Common checks include verifying that required fields are not null, confirming row counts fall within expected ranges, and checking that values fall within valid bounds."},
    {"doc_id": 10, "title": "Chunking Strategies for RAG Systems", "text": "Chunking is the process of splitting long documents into smaller pieces before generating embeddings for retrieval. The size of each chunk matters: chunks that are too large can dilute relevance, while chunks that are too small may lose important context. A common strategy is to use a fixed word or token count per chunk, with some overlap between consecutive chunks to preserve context across boundaries."},
    {"doc_id": 11, "title": "Claude and Anthropic Overview", "text": "Claude is a family of large language models developed by Anthropic, designed with a strong focus on safety and helpfulness. Claude models can be accessed through an API, as well as through consumer products like Claude.ai, and through developer tools such as Claude Code."},
    {"doc_id": 12, "title": "Batch vs Streaming Data Processing", "text": "Batch processing handles data in large groups at scheduled intervals, such as once an hour or once a day, making it well suited for reporting and historical analysis. Streaming processing, by contrast, handles data continuously as it arrives, enabling near real-time insights for use cases like fraud detection or live dashboards."},
    {"doc_id": 13, "title": "Understanding Data Lakehouses", "text": "A data lakehouse combines the low-cost, flexible storage of a data lake with the data management and transactional features typically found in a data warehouse. This architecture allows organizations to store structured, semi-structured, and unstructured data together while still supporting reliable SQL analytics and machine learning workloads."},
    {"doc_id": 14, "title": "Common ETL Pipeline Failures", "text": "ETL pipelines can fail for many reasons, including schema changes in source systems, unexpected null values, duplicate records, and network timeouts during data transfer. A robust pipeline anticipates these failures by including validation steps, retry logic, and clear logging at each stage."},
    {"doc_id": 15, "title": "Introduction to Prompt Engineering", "text": "Prompt engineering is the practice of crafting inputs to a language model in order to get more accurate, relevant, or structured outputs. Effective prompts often include clear instructions, relevant context, and examples of the desired output format. Techniques such as chain-of-thought prompting can improve performance on complex tasks."},
]

CHUNK_SIZE_WORDS = 60
OVERLAP_WORDS = 15


def chunk_text(doc_id: int, title: str, text: str) -> list:
    words = text.split()
    chunks = []
    start, idx = 0, 0
    while start < len(words):
        end = min(start + CHUNK_SIZE_WORDS, len(words))
        chunks.append({
            "chunk_id": f"{doc_id}_{idx}",
            "doc_id": doc_id,
            "title": title,
            "chunk_text": " ".join(words[start:end]),
        })
        idx += 1
        if end == len(words):
            break
        start += CHUNK_SIZE_WORDS - OVERLAP_WORDS
    return chunks


# ---------------------------------------------------------------------------
# Step 2: Embedding -- turn each chunk into a vector via Azure AI Foundry
# ---------------------------------------------------------------------------

def get_embedding(text: str) -> np.ndarray:
    response = embed_client.embed(input=[text], model=EMBEDDING_DEPLOYMENT)
    return np.array(response.data[0].embedding)


def build_index(chunks: list) -> list:
    """Embeds every chunk once and stores the vector alongside it -- this
    in-memory list IS the 'vector database' in a from-scratch build."""
    index = []
    for chunk in chunks:
        vector = get_embedding(chunk["chunk_text"])
        index.append({**chunk, "embedding": vector})
    return index


# ---------------------------------------------------------------------------
# Step 3: Retrieval -- manual cosine similarity, no vector DB library
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve(query: str, index: list, k: int = 3) -> list:
    query_vector = get_embedding(query)
    scored = [
        {**item, "score": cosine_similarity(query_vector, item["embedding"])}
        for item in index
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]


# ---------------------------------------------------------------------------
# Step 4: Generation -- grounded answer using only the retrieved context
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the
provided context. If the context does not contain the answer, say so clearly
rather than guessing. Cite which source (by title) supports your answer."""


def generate_answer(query: str, retrieved_chunks: list) -> str:
    context = "\n\n".join(
        f"[Source: {c['title']}]\n{c['chunk_text']}" for c in retrieved_chunks
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    response = chat_client.complete(
        messages=[SystemMessage(content=SYSTEM_PROMPT), UserMessage(content=user_prompt)],
        model=CHAT_DEPLOYMENT,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_knowledge_base() -> list:
    all_chunks = []
    for article in ARTICLES:
        all_chunks.extend(chunk_text(article["doc_id"], article["title"], article["text"]))
    print(f"Chunked {len(ARTICLES)} articles into {len(all_chunks)} chunks.")
    print("Embedding all chunks via Azure AI Foundry...")
    return build_index(all_chunks)


def ask(query: str, index: list, k: int = 3, verbose: bool = True) -> dict:
    retrieved = retrieve(query, index, k=k)
    answer = generate_answer(query, retrieved)

    if verbose:
        print(f"\nQuestion: {query}")
        print("\nRetrieved chunks:")
        for r in retrieved:
            print(f"  [{r['score']:.3f}] {r['title']} -- {r['chunk_text'][:80]}...")
        print(f"\nAnswer:\n{answer}")

    return {"query": query, "retrieved": retrieved, "answer": answer}


if __name__ == "__main__":
    index = build_knowledge_base()

    questions = [
        "What is the medallion architecture and what are its three layers?",
        "How does RAG reduce hallucinations?",
        "What's the difference between batch and streaming processing?",
    ]
    for q in questions:
        ask(q, index)
        print("\n" + "=" * 70)
