# Demo FAQ

This document helps demo the RAG app. It contains facts that the app can retrieve when you ask questions after uploading its DOCX/PDF version.

## About The App
- The application is a minimal Retrieval Augmented Generation (RAG) demo.
- It uses MongoDB as the database for storing document chunks and their embeddings.
- It uses OpenAI to generate embeddings and to answer questions via a chat model.
- It uses a simple Flask server with an inline HTML UI.

## Setup
- The database runs in Docker using the `mongo:4.4` image with `platform: linux/amd64` for Apple Silicon compatibility.
- A script `setup.sh` boots MongoDB and seeds some sample documents.
- The `.env` file stores sensitive values, including the OpenAI API key.

## Models
- The embedding model is `text-embedding-3-small` by default.
- The chat model is `gpt-4o-mini` by default.

## Usage
- You can upload PDF and DOCX files.
- The app splits documents into chunks before computing embeddings.
- During search, the query is embedded and compared to stored vectors using cosine similarity.
- The top matches are added as context for the answer generation.
- If the answer cannot be found in the uploaded documents, the app can fall back to a general LLM answer.

## Tips
- Keep documents reasonably short for a quick demo.
- Ask specific questions that are likely to match a chunk.
- Example questions:
  - "Which database is used by the app?"
  - "Which model is used for embeddings?"
  - "How do I start the database?"
  - "What file types are supported for upload?"
