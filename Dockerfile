# syntax=docker/dockerfile:1.7

# Two sizes, one Dockerfile:
#   docker build .                                      -> lean (default): no reranker, smallest image
#   docker build --build-arg INSTALL_EXTRAS=rerank .     -> full: adds the cross-encoder reranker
#
# The reranker (sentence-transformers + transformers + torch) is ~650-700MB
# installed on its own — by far the single heaviest dependency in this
# project, for one optional relevance improvement. Leaving it out entirely is
# the single biggest lever for image size; RRF's own ordering is still used
# when it's off, so retrieval degrades gracefully, not silently.
#
# Base is debian-slim, not alpine: several dependencies here (numpy, lxml,
# chroma-hnswlib, scipy) ship manylinux wheels but not musllinux ones, so
# alpine would compile them from source on every build — slower, and a real
# risk of a C-extension subtly misbehaving under musl libc. Slim keeps the
# wheel-based install while still staying a few hundred MB smaller than the
# full python image.

ARG PYTHON_VERSION=3.11-slim

# ---------------------------------------------------------------------------
# builder — compiles/installs into a venv. The compiler toolchain and pip's
# download cache never make it into the final image; only the venv does.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY pyproject.toml ./
COPY app ./app

ARG INSTALL_EXTRAS=""
# --extra-index-url pulls CPU-only torch wheels (only resolved at all when
# the rerank extra is requested) — the default PyPI torch wheel bundles CUDA
# libraries this image has no GPU to use, multiple times the size of the
# CPU-only build for zero benefit in a CPU-only container.
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ -n "$INSTALL_EXTRAS" ]; then \
        pip install --extra-index-url https://download.pytorch.org/whl/cpu ".[$INSTALL_EXTRAS]"; \
    else \
        pip install .; \
    fi

# ---------------------------------------------------------------------------
# final — just the venv and the app source. No compiler, no pip cache, no
# dev/test tooling, no Streamlit (a separate, optional frontend process).
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION} AS final

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    RERANK_ENABLED=false

RUN groupadd -r pke && useradd -r -g pke -d /app pke

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY app ./app
COPY docker-entrypoint.sh ./

RUN mkdir -p /app/data && chown -R pke:pke /app && chmod +x docker-entrypoint.sh

USER pke

EXPOSE 8000
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
