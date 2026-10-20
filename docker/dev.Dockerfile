# Development/CI image for rampgate. Nothing is installed on the host;
# every check runs in here via ./dev.
FROM ghcr.io/astral-sh/uv:0.9.5-bookworm-slim

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_CACHE_DIR=/cache/uv \
    UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=only-managed \
    HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1

RUN uv python install 3.11 3.12 3.13 \
    && chmod -R a+rX /opt/uv-python \
    && mkdir -p /cache/uv && chmod -R 1777 /cache

WORKDIR /work
