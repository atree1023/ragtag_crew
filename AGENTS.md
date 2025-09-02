# Overview

- Searchable, up to date documentation and code examples are critical context for AI coding agents
- This project will have toolsets for creating, updating and accessing vector databases with that documentation and examples
- The vectordb will be Pinecone
- One tool will handle collection and upserting
- Metadata will include the document name, document url, date of collection and section of the document
- The insert tool will run after doc updates and remove the old version
- The third will be the MCP tools for agent and model access to those DBs.
- Langchain used for text splitting and chunking
