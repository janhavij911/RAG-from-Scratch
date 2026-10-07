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
