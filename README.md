# RAG From Scratch

A Retrieval-Augmented Generation system built without any RAG framework — no LangChain, no managed vector database. Every piece is implemented by hand: chunking, embedding, cosine similarity search, and grounded generation — using models deployed in Azure AI Foundry.

## Why "from scratch"

Most RAG tutorials wrap everything in a framework, which is great for speed but hides what's actually happening. This project implements each piece explicitly:
- **Chunking**: a manual sliding-window function, not a library call
- **Embedding**: direct calls to an Azure AI Foundry embedding deployment
- **Vector storage**: a plain Python list of dicts — no vector database
- **Retrieval**: cosine similarity computed by hand with NumPy
- **Generation**: an explicit, visible prompt template, not a hidden framework default

## Architecture
15 articles
│
▼
Chunking (60 words, 15-word overlap) — same strategy as the Databricks Gold layer project
│
▼
Embedding (Azure AI Foundry — text-embedding-3-small)
│
▼
In-memory vector store (list of {text, embedding})
│
▼
Query → embed → cosine similarity → top-k chunks
│
▼
Generation (Azure AI Foundry — gpt-5.6-sol), grounded in retrieved context, with source citation

Ask a question about the knowledge base:

Common ETL Pipeline failures
Retrieved Chunks (by cosine similarity)
[0.763] Common ETL Pipeline Failures

ETL pipelines can fail for many reasons, including schema changes in source systems, unexpected null values, duplicate records, and network timeouts during data transfer. A robust pipeline anticipates these failures by including validation steps, retry logic, and clear logging at each stage.

[0.388] What is Data Engineering

Data engineering is the practice of designing and building systems that collect, store, and analyze data at scale. Data engineers build pipelines that transform raw data into a usable format for analysts and data scientists. Core responsibilities include building ETL and ELT pipelines, managing data warehouses and lakes, and ensuring data quality and reliability.

[0.345] Why Data Quality Checks Matter

Data quality checks are automated validations that catch problems in a dataset before they reach downstream users or systems. Common checks include verifying that required fields are not null, confirming row counts fall within expected ranges, and checking that values fall within valid bounds.

Grounded Answer
Common ETL pipeline failures include:

Schema changes in source systems
Unexpected null values
Duplicate records
Network timeouts during data transfer
Robust pipelines mitigate these risks with validation steps, retry logic, and clear logging at each stage.

Source: Common ETL Pipeline Failures

## Running It

```bash
pip install -r rag_requirements.txt
```

Create a `.env` file:

AZURE_FOUNDRY_ENDPOINT=https://your-resource.services.ai.azure.com/models
AZURE_FOUNDRY_KEY=your-key
AZURE_FOUNDRY_CHAT_DEPLOYMENT=gpt-5.6-sol
AZURE_FOUNDRY_EMBEDDING_DEPLOYMENT=text-embedding-3-small


```bash
python rag_from_scratch.py      # CLI demo, 3 sample questions
streamlit run rag_app.py        # interactive UI with similarity scores
```

## Tech Stack
Azure AI Foundry (gpt-5.6-sol, text-embedding-3-small), Python, NumPy, Streamlit

## Part of a Connected Portfolio
Reuses the exact chunking strategy from the Databricks Bronze-Silver-Gold pipeline project — this is that pipeline's natural next stage, now wired to real embeddings and retrieval.
