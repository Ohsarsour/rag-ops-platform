# Embedding Service — multi-stage build (Req 9.1, 9.2, 9.7).
#
# Stage 1 (builder): installs build toolchains and Python dependencies into an
# isolated virtualenv. Stage 2 (runtime): copies only that virtualenv plus the
# application code, so the final image carries no compilers/build deps.
#
# Why multi-stage at all? It keeps the shipped image small and
# the compilers and headers needed to BUILD dependencies never make it
# into the image we actually RUN. 
#
# Build context is this service's own directory, e.g.:
#   docker build -t ragops-embedding services/embedding

# ---------- Stage 1: builder ----------
# Pin a specific base (python:3.11-slim) so builds are reproducible and we meet
# the Python 3.11+ target (Req 9.7). "slim" drops docs/locales we don't need.
FROM python:3.11-slim AS builder

# Install a C toolchain only in the builder: some Python wheels (native
# extensions) compile from source when no prebuilt wheel exists. `--no-install-
# recommends` skips optional extras and `rm -rf .../lists` deletes the apt cache
# in the SAME layer so it never bloats the image. 
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Build into an isolated virtualenv at a known path so the runtime stage can
# copy the whole dependency tree as one self-contained directory.
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV"
# Put the venv first on PATH so `pip`/`python` below resolve to it automatically.
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy ONLY requirements.txt before the app code so Docker's layer cache reuses
# the (slow) dependency install whenever app code changes but deps don't.
# order Dockerfile steps from least- to most-frequently changing.
COPY requirements.txt .
# `--no-cache-dir` keeps pip's download cache out of the image; upgrading pip
# first avoids resolver bugs in older versions. Pinned versions live in
# requirements.txt so the build is deterministic (Req 9.3).
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------- Stage 2: runtime ----------
# Start from a fresh, clean base — nothing from the builder carries over except
# what we explicitly COPY, so build toolchains are left behind (Req 9.2).
FROM python:3.11-slim AS runtime

# PYTHONUNBUFFERED: logs stream out immediately (critical for container logs /
# debugging). PYTHONDONTWRITEBYTECODE: skip .pyc clutter in an immutable image.
# Re-declare the venv on PATH so the copied dependencies are used at runtime.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Bring over just the prepared virtualenv from the builder — no compilers, no
# apt cache, no build artifacts. This is the line that makes the image lean.
COPY --from=builder /opt/venv /opt/venv

# Set the working directory so `app.main:app` resolves relative to /app.
WORKDIR /app

# Create and switch to a non-root user. Running as root inside a container is a
# common security finding: if the app is compromised, the attacker isn't root.
#least privilege — never run app containers as root."
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Copy the application code last (it changes most often, so it busts the fewest
# cached layers) and own it as the non-root user.
COPY --chown=appuser:appuser app ./app

# Document the port the service listens on. EXPOSE is metadata only,  it doesn't
# publish the port; compose/`-p` does that — but it signals intent to operators.
EXPOSE 8000

# Exec-form CMD (JSON array) so uvicorn is PID 1 and receives SIGTERM directly,
# giving graceful shutdown. Binding 0.0.0.0 makes it reachable from outside the
# container. Matches the FastAPI app object at app/main.py: `app`.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
