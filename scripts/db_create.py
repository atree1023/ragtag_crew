"""Create the Pinecone index for ragtag_crew if it does not already exist.

Environment:
- PINECONE_API_KEY: API key for Pinecone.

Usage:
- Run as a script or import; it is idempotent and safe to re-run.
"""

import os

from pinecone import IndexEmbed, Pinecone

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "ragtag-db"

if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed=IndexEmbed(
            model="llama-text-embed-v2",
            field_map={"text": "chunk_content"},
            metric="cosine",
        ),
    )
