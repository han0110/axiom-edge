# Multi-stage build for edge-manager and edge-worker binaries

ARG BUILDER_IMAGE=nvidia/cuda:13.1.0-devel-ubuntu24.04
ARG RUNTIME_IMAGE=nvidia/cuda:13.1.0-runtime-ubuntu24.04
ARG VCS_REF=unknown
ARG OPENVM_REV=unknown
ARG STARK_BACKEND_REV=unknown
ARG JEMALLOC_SYS_WITH_MALLOC_CONF=retain:true,background_thread:true,metadata_thp:always,dirty_decay_ms:10000,muzzy_decay_ms:10000,abort_conf:true

# Tools stage: Install cargo tools once and reuse
FROM ${BUILDER_IMAGE} AS tools

ARG CARGO_BUILD_JOBS=32

# Use AWS mirrors for apt for faster downloads from EC2
RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/us-east-1.ec2.archive.ubuntu.com\/ubuntu\//g' /etc/apt/sources.list.d/ubuntu.sources || \
    sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/us-east-1.ec2.archive.ubuntu.com\/ubuntu\//g' /etc/apt/sources.list || true

# Install build dependencies with cache mounts
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    clang \
    curl \
    git \
    openssh-client \
    pkg-config \
    libssl-dev \
    ca-certificates

# Install Rust, only if not already present in the base image
RUN if ! command -v rustc >/dev/null 2>&1; then \
    echo "rustc not found, installing Rust with rustup..."; \
    curl https://sh.rustup.rs -sSf | bash -s -- -y; \
    else \
    echo "rustc found, skipping installation."; \
    fi
ENV PATH="/root/.cargo/bin:${PATH}"

ENV CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS}
# Install cargo tools - this will be cached as a Docker layer
# Use pre-built sccache binary to avoid compilation time
RUN cargo install cargo-chef && \
    ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
    SCCACHE_ARCH="x86_64-unknown-linux-musl"; \
    elif [ "$ARCH" = "aarch64" ]; then \
    SCCACHE_ARCH="aarch64-unknown-linux-musl"; \
    else \
    echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    echo "Downloading sccache for $SCCACHE_ARCH..." && \
    curl -fsSL "https://github.com/mozilla/sccache/releases/download/v0.12.0/sccache-v0.12.0-${SCCACHE_ARCH}.tar.gz" | \
    tar -xzf - -C /tmp && \
    mkdir -p /root/.cargo/bin && \
    mv /tmp/sccache-v0.12.0-${SCCACHE_ARCH}/sccache /root/.cargo/bin/sccache && \
    chmod +x /root/.cargo/bin/sccache && \
    sccache --version

# Set up sccache
ENV RUSTC_WRAPPER=sccache SCCACHE_DIR=/sccache

# Base image with build tools - inherits from tools stage
FROM tools AS base

# Install remaining build dependencies with cache mounts
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    protobuf-compiler

# Ensure github.com is a known host to avoid interactive prompts
RUN mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
ENV CARGO_NET_GIT_FETCH_WITH_CLI=true

# Pre-install the specified toolchain if provided, so it's cached in the Docker layer
ARG TOOLCHAIN_VERSION
RUN if [ -n "$TOOLCHAIN_VERSION" ]; then \
    echo "Installing toolchain: ${TOOLCHAIN_VERSION}"; \
    rustup toolchain install "${TOOLCHAIN_VERSION##+}" --profile minimal --component rust-src; \
    fi

WORKDIR /app

# Planner stage: Generate dependency recipe
FROM base AS planner
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    cargo chef prepare --recipe-path recipe.json

# Builder: Build dependencies and application
FROM base AS builder

ARG CARGO_BUILD_JOBS=32
ENV CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS}
ARG PROFILE=release
ARG FEATURES=cuda,jemalloc,parallel
ARG CUDA_ARCH
ARG TARGET_CPU=native
ARG TOOLCHAIN_VERSION
ARG JEMALLOC_SYS_WITH_MALLOC_CONF

# Pass CUDA_ARCH to the build script so it doesn't try to run nvidia-smi
ENV CUDA_ARCH=${CUDA_ARCH}
ENV RUSTFLAGS="-Ctarget-cpu=${TARGET_CPU}"
ENV JEMALLOC_SYS_WITH_MALLOC_CONF=${JEMALLOC_SYS_WITH_MALLOC_CONF}

# Build dependencies using cargo-chef
COPY --from=planner /app/recipe.json recipe.json
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    cargo ${TOOLCHAIN_VERSION} chef cook --profile "$PROFILE" -p edge-worker --features "$FEATURES" --recipe-path recipe.json && \
    cargo ${TOOLCHAIN_VERSION} chef cook --profile "$PROFILE" -p edge-manager --recipe-path recipe.json

# Build the application, reusing dependency cache
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    set -eu; \
    cargo ${TOOLCHAIN_VERSION} fetch --locked; \
    cargo ${TOOLCHAIN_VERSION} build --profile "$PROFILE" -p edge-manager; \
    cargo ${TOOLCHAIN_VERSION} build --profile "$PROFILE" -p edge-worker --features "$FEATURES"

# Runtime: Final slim image
FROM ${RUNTIME_IMAGE}

ARG VCS_REF
ARG OPENVM_REV
ARG STARK_BACKEND_REV
ARG JEMALLOC_SYS_WITH_MALLOC_CONF=retain:true,background_thread:true,metadata_thp:always,dirty_decay_ms:10000,muzzy_decay_ms:10000,abort_conf:true

# Use AWS mirrors for apt for faster downloads from EC2
RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/us-east-1.ec2.archive.ubuntu.com\/ubuntu\//g' /etc/apt/sources.list.d/ubuntu.sources || \
    sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/us-east-1.ec2.archive.ubuntu.com\/ubuntu\//g' /etc/apt/sources.list || true

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libssl3 \
    ca-certificates \
    tini \
    gcc \
    binutils \
    libc6-dev

WORKDIR /app

LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL com.axiom.openvm_rev="${OPENVM_REV}"
LABEL com.axiom.stark_backend_rev="${STARK_BACKEND_REV}"

# Re-declare the PROFILE argument to make it available in this stage
ARG PROFILE=release

# Copy the built binaries from the builder stage. `convert_fixtures` derives a
# deployment's proving keys and vmexe from a guest ELF, which a host runs once
# before starting its workers.
COPY --from=builder /app/target/$PROFILE/edge-manager .
COPY --from=builder /app/target/$PROFILE/edge-worker .
COPY --from=builder /app/target/$PROFILE/convert_fixtures .

# Copy config files
COPY config/ ./config/

# Copy the entrypoint script
COPY docker/docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x docker-entrypoint.sh

ENV RUST_BACKTRACE=full
ENV JEMALLOC_SYS_WITH_MALLOC_CONF=${JEMALLOC_SYS_WITH_MALLOC_CONF}
# Segmentation defaults (override at runtime to tune proving performance)
# OPENVM_MAX_SEGMENT_LENGTH: max trace height per segment (default: 2^22 = 4194304)
# OPENVM_MAX_SEGMENT_MEMORY: max memory per segment in bytes (default: 16106127360 = 15GiB)
# These are only applied if set; otherwise OpenVM library defaults are used.

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["edge-worker"]
