---
applyTo: "**"
---

# GitHub Copilot - Agent Instructions

## Tools Overview

Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.

### Langchain, Langgraph, and Langchain Text Splitter

- For any questions pertaining to Langchain, Langgraph, or Langchain Text Splitter, utilize the `langgraph-docs-mcp` MCP server.
- Use the `list_doc_sources` tool to obtain the available `llms.txt` file.
- Use the `fetch_docs` tool to access the contents of `llms.txt`.
- Before any significant tool call, state one line: purpose + minimal inputs.
- Examine the URLs listed in `llms.txt` and assess their relevance to the given question.
- Carefully consider the specific details or requirements of the inquiry.
- For any URL in `llms.txt` that seems pertinent, use the `fetch_docs` tool to retrieve relevant documentation.

### Pinecone

- For all questions related to Pinecone, utilize the `pinecone` MCP server and the `search-docs` tool.
