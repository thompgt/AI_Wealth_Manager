# syntax=docker/dockerfile:1

# Two stages. The builder holds pip, the compilers psycopg2 and numpy want,
# and the wheel cache; the runtime holds the interpreter, the installed
# packages and the source. The reason to split is not image size -- it is that
# a build toolchain in a running container is a toolchain an attacker who gets
# code execution can use.
#
# One image serves both processes. The API and the job worker share every
# import -- the graph, the agents, the services, the models -- so building
# them separately would produce two nearly identical images that could drift
# apart in dependency versions, which is precisely the failure the exact pins
# in requirements.txt exist to prevent. The command differs; the code does not.

# ---------------------------------------------------------------------------
FROM python:3.13-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# build-essential for any sdist that still needs a compiler; libpq-dev for
# psycopg2. Both stay in this stage.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# A virtualenv rather than --user or the system site-packages: it is one
# self-contained directory to copy into the runtime stage, with no ambiguity
# about which interpreter owns which package.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copied alone, before the source, so editing an agent does not reinstall
# pandas. The pins make this layer genuinely reproducible.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

# libpq5 is the runtime half of libpq-dev -- psycopg2 links against it and
# fails to import without it. curl is here for the container healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libpq5 curl \
 && rm -rf /var/lib/apt/lists/*

# Non-root. This process reaches the internet, renders text the firm did not
# write into LLM prompts, and holds client PII. Running it as uid 0 means any
# code-execution bug is a root-in-container bug, and with a bind mount, often
# root on the host filesystem too.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app . .

# Writable state that must not live in the image layer: the SQLite checkpoint
# file when Postgres is not configured, and anything else the process writes.
# Declared as a volume so a container restart does not silently discard a
# paused human-approval run.
RUN mkdir -p /app/data && chown app:app /app/data
VOLUME ["/app/data"]

USER app

EXPOSE 8000

# Readiness, not liveness: an orchestrator that routes traffic on liveness
# sends requests to a replica whose database is unreachable. `/ready` answers
# the question that actually gates traffic.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/ready || exit 1

# The API only. JOB_WORKER_ENABLED=false on this replica and a separate
# container running `python worker.py` keeps a long graph run from competing
# with request handling for the same GIL, and lets the two scale apart.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
