# syntax=docker/dockerfile:1
#
# Build vllm v0.15.1 with free-threaded Python 3.14t using uv.
#
# Usage:
#   docker build --build-arg TORCH_CUDA_ARCH_LIST="7.5" -t vllm-freethreaded .
#   docker run --gpus all -e PYTHON_GIL=0 vllm-freethreaded \
#       python -c "import vllm; print(vllm.__version__)"
#
# The build uses a multi-stage approach. Final stage produces a minimal image
# based on nvidia/cuda base.  Build with --squash for a single-layer image.
#
# Written with help from Claude Opus 4.6.

# ---------------------------------------------------------------------------
# deps — System packages + uv + Python + Rust
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

# CUDA env vars
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update && apt-get install -y --no-install-recommends \
        git \
        ccache \
        numactl \
        libnuma-dev \
        g++ \
        curl \
        pkg-config \
        libssl-dev \
        protobuf-compiler \
        ca-certificates

# Install Rust (cached across builds; not stored in image layers)
RUN --mount=type=cache,target=/root/.rustup \
    --mount=type=cache,target=/root/.cargo \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.10.2 /uv /uvx /bin/

# Create a free-threaded Python 3.14t venv
RUN uv venv /opt/venv --python cpython-3.14t
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"


# ---------------------------------------------------------------------------
# python-deps — Install pip dependencies
# ---------------------------------------------------------------------------
FROM base AS python-deps

# Install PyTorch + torchaudio with CUDA 12.8 index
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_HTTP_TIMEOUT=90 uv pip install \
        "torch>=2.10.0" \
        "torchaudio>=2.10.0" \
        --extra-index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies from pyproject.toml
COPY pyproject.toml /tmp/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r /tmp/pyproject.toml

# ---------------------------------------------------------------------------
# build — Clone repos + build from source
# ---------------------------------------------------------------------------
FROM python-deps AS build

ARG TORCH_CUDA_ARCH_LIST="7.5"
ARG MAX_JOBS="4"
ARG NVCC_THREADS="4"

WORKDIR /app

# Set git identity (needed for git am in clone-repos.py)
RUN git config --global user.email "build@docker.example.com" \
    && git config --global user.name "Docker Build"

# Copy clone infrastructure
COPY clone-repos.py git-repos.txt ./
COPY patches/ patches/

# Clone all required repos
RUN python clone-repos.py \
        --repo vllm \
        --repo flash-attention \
        --repo harmony \
        --repo safetensors \
        --repo tokenizers

# Build safetensors from source (non-editable — goes into site-packages)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.rustup \
    --mount=type=cache,target=/root/.cargo \
    uv pip install safetensors/safetensors/bindings/python \
        --no-build-isolation --no-deps

# Build tokenizers from source (non-editable)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.rustup \
    --mount=type=cache,target=/root/.cargo \
    uv pip install tokenizers/tokenizers/bindings/python \
        --no-build-isolation --no-deps

# Build harmony (openai-harmony) from source (non-editable)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.rustup \
    --mount=type=cache,target=/root/.cargo \
    uv pip install harmony/harmony \
        --no-build-isolation --no-deps

# Build vllm from source (editable — source tree needed at runtime)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.ccache \
    CCACHE_DIR=/root/.ccache \
    VLLM_FLASH_ATTN_SRC_DIR=/app/flash-attention/flash-attention \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    MAX_JOBS="${MAX_JOBS}" \
    NVCC_THREADS="${NVCC_THREADS}" \
    CC="ccache gcc" \
    CXX="ccache g++" \
    CMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    uv pip install -e vllm/vllm -v \
        --no-build-isolation --no-deps

# ---------------------------------------------------------------------------
# staging — Assemble only the files needed at runtime
# ---------------------------------------------------------------------------
FROM build AS staging

RUN set -eux \
    # --- Create staging directory structure --- \
    && mkdir -p /staging/usr/lib/x86_64-linux-gnu \
                /staging/opt \
                /staging/root/.local/share/uv \
                /staging/app \
    # --- CUDA-related system shared libs (cublas, cudnn, nccl, …) --- \
    # The base nvidia/cuda image already provides /usr/local/cuda-12.8 \
    # runtime libs; we only need the extra libraries installed via apt \
    # in the devel image (cublas, cudnn, nccl, etc.). \
    && for lib in libcublas libcublasLt libcufft libcurand libcusolver \
                  libcusparse libnccl libnvrtc libnvJitLink libcudnn \
                  libnvToolsExt; do \
         for f in /usr/lib/x86_64-linux-gnu/${lib}*; do \
           [ -e "$f" ] && cp -a "$f" /staging/usr/lib/x86_64-linux-gnu/ || true; \
         done; \
       done \
    # --- Python venv + the uv-managed Python it symlinks into --- \
    && cp -a /opt/venv /staging/opt/venv \
    && cp -a /root/.local/share/uv/python /staging/root/.local/share/uv/python \
    # --- vllm source (editable install needs this at runtime) --- \
    && cp -a /app/vllm /staging/app/vllm \
    && rm -rf /staging/app/vllm/vllm/.deps

# ---------------------------------------------------------------------------
# runtime — Minimal production image
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.1-base-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Runtime system dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update && apt-get install -y --no-install-recommends \
        vim-tiny \
        less \
        git \
        numactl \
        libgomp1 \
        ca-certificates \
        gcc \
        libc6-dev

# All runtime files in a single COPY layer
COPY --from=staging /staging/ /

# Refresh the dynamic linker cache
RUN ldconfig

# Environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHON_GIL=0

WORKDIR /app
