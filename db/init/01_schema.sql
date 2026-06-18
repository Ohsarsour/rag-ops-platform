-- 01_schema.sql — Postgres + pgvector schema for the RAG Ops Platform (Phase 1)
--
-- This script is mounted into /docker-entrypoint-initdb.d/ and run automatically
-- by Postgres on the first boot of an empty data directory. A named Docker volume
-- (pgdata) backs the data directory so chunks/vectors survive restarts (Req 5.8).
--
-- Requirement traceability: 5.1, 5.2, 5.3, 5.4, 5.6 (with 5.5/5.7 enforced by the
-- vector(384) column type, and 5.8 handled by the compose volume).

-- Req 5.1: the Data_Store is a PostgreSQL instance with the pgvector extension enabled.
-- Req 5.3: if pgvector is unavailable, CREATE EXTENSION errors here and init halts,
--          so the database never comes up in a half-configured state.
CREATE EXTENSION IF NOT EXISTS vector;

-- documents: one row per ingested Source_Document.
-- Supports Req 5.2 (source path/name metadata) and the ingestion chunk_count
-- bookkeeping (a document with zero chunks is still recorded — Req 3.6).
CREATE TABLE IF NOT EXISTS documents (
    id          BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_path TEXT        NOT NULL,
    source_name TEXT        NOT NULL,
    chunk_count INTEGER     NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- chunks: one row per Chunk, with its text, embedding, and metadata.
-- Req 5.2: stores chunk text, a 384-dim embedding, source document id, source
--          path, source name, and a zero-based non-negative chunk position index.
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id BIGINT      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    source_path TEXT        NOT NULL,      -- denormalized for fast retrieval response (Req 6.3)
    source_name TEXT        NOT NULL,
    chunk_index INTEGER     NOT NULL CHECK (chunk_index >= 0),  -- Req 5.2 (zero-based, non-negative)
    content     TEXT        NOT NULL,
    -- Req 5.4: fixed 384-dim vector column. The vector(384) type makes Postgres
    -- itself reject any insert (Req 5.5) or query vector (Req 5.7) of the wrong size,
    -- so dimension enforcement lives in the database, not application code.
    embedding   vector(384) NOT NULL
);

-- Req 5.6: HNSW index using cosine distance (vector_cosine_ops) so the Retrieval_Service
-- can run fast cosine similarity search (`embedding <=> query`). HNSW builds on an empty
-- table (unlike IVFFlat) and gives high recall at low latency for our sub-10M corpus.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Speeds up document-scoped lookups and the ON DELETE CASCADE foreign-key checks.
CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks(document_id);
