# LlamaIndexHelper
A chatbot specifically designed to answer questions about RAG and LlamaIndex.

## Data
- Craw from https://www.llamaindex.ai/blog

## Pipelines
[Basic pipeline RAG](./assets/rag_basic_pipeline.png)

- Crawl and clean html web pages
- Build local vector database with chromadb
- Build document summary index
- Build basic RAG with pipeline with Llama-2 model

## Results
<!-- - **Question**: `"What are key features of llama-agents?"`
- **Answer**:
![Response 1](./assets/demo_response_1.png) -->

## TODO
- [x] Enhance querying stage with SummaryIndex and Router Engine
- [ ] Evaluate RAG pipeline