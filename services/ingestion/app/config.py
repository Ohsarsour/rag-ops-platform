"""Configuration for the Ingestion Service.

Everything that could differ between machines or environments (where the
Embedding Service lives, how to reach the database) is read from environment
variables here — nothing is hardcoded. This mirrors the Embedding Service's
``config.py`` pattern and satisfies the design's "Common conventions" rule that
all services read URLs/DSNs from the environment so the same image runs locally
and in Docker without code changes (Req 8.3).

If you are new to this: an "environment variable" is just a named value the
operating system (or Docker Compose) hands to the program when it starts, e.g.
``EMBEDDING_URL=http://embedding:8000``. Reading them in one place means the
rest of the code never has to care *where* a value came from.
"""
from __future__ import annotations

import os

# Base URL of the Embedding Service (Req 4.1/4.2). In Docker Compose the
# services reach each other by service name, so this defaults to the compose
# service name "embedding" on its port. Override with the EMBEDDING_URL env var.
#
# We POST chunk texts to "{EMBEDDING_URL}/embed/batch" to get their vectors.
EMBEDDING_URL: str = os.environ.get("EMBEDDING_URL", "http://embedding:8000")

# PostgreSQL connection string (DSN) for the Data_Store. A DSN ("Data Source
# Name") is a single string that tells the database driver how to connect:
# user, password, host, port, and database name. Example:
#   postgresql://raguser:ragpass@postgres:5432/ragdb
# There is no safe universal default for a password-protected database, so we
# fall back to a local-development DSN only; real deployments set DATABASE_URL.
DATABASE_URL: str = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/postgres"
)

# How long (in seconds) we wait for the Embedding Service before giving up.
# Req 4.7 fixes this at 30 seconds: if a connection cannot be made or no
# response arrives within this window, ingestion fails with HTTP 502.
EMBEDDING_TIMEOUT_SECONDS: float = 30.0

# The embedding dimensionality every stored vector must have. The database
# column is vector(384), so this is informational/validation only here; the
# database is the real enforcement point (Req 5.4/5.5).
EMBEDDING_DIMENSIONS: int = 384
