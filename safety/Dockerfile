# Evaluation image for the 2048 competition (Python + Julia, CPU).
#
#   docker build -t 2048-eval:latest .
#   docker build -t 2048-eval:latest --build-arg WITH_TENSORFLOW=1 .
#
# The harness runs this image with --network none, a read-only root filesystem,
# a non-root user and resource limits (see harness.py). Everything participants
# may import must be installed HERE, at build time.
#
# Pin things: use a fixed base digest (docker pull python:3.12-slim-bookworm,
# then FROM python@sha256:...), and freeze the Julia Manifest (see bottom).
FROM python:3.12-slim-bookworm

ARG JULIA_VERSION=1.10.5
ARG WITH_TENSORFLOW=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---- Julia ----------------------------------------------------------------
RUN set -eux; \
    arch="$(uname -m)"; \
    case "$arch" in \
      x86_64)  jdir=x64;     jarch=x86_64 ;; \
      aarch64) jdir=aarch64; jarch=aarch64 ;; \
      *) echo "unsupported architecture: $arch"; exit 1 ;; \
    esac; \
    minor="${JULIA_VERSION%.*}"; \
    curl -fsSL "https://julialang-s3.julialang.org/bin/linux/${jdir}/${minor}/julia-${JULIA_VERSION}-linux-${jarch}.tar.gz" \
      | tar -xz -C /opt; \
    ln -s "/opt/julia-${JULIA_VERSION}/bin/julia" /usr/local/bin/julia

# ---- Python packages (extend as the jury decides) --------------------------
RUN pip install --no-cache-dir \
        numpy==2.0.2 scipy==1.14.1 pandas==2.2.3 scikit-learn==1.5.2 numba==0.60.0 \
    && pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && if [ "$WITH_TENSORFLOW" = "1" ]; then pip install --no-cache-dir tensorflow-cpu==2.18.0; fi

# ---- Julia packages, precompiled at build time ------------------------------
# JSON3 is required by shim.jl. Precompiling here means compile time is not
# charged to (or randomly dodged by) any agent's 30-minute budget.
ENV JULIA_DEPOT_PATH=/opt/julia_depot
RUN mkdir -p /opt/julia_env \
    && julia --project=/opt/julia_env -e 'using Pkg; Pkg.add(["JSON3", "Flux", "StatsBase", "Distributions"]); Pkg.precompile()' \
    && chmod -R a+rX /opt/julia_env /opt/julia_depot

# To freeze exact Julia versions after the first build:
#   docker run --rm 2048-eval:latest cat /opt/julia_env/Manifest.toml > Manifest.toml
# then add before the RUN above:  COPY Manifest.toml /opt/julia_env/Manifest.toml

# Mount points used by the harness (rootfs is read-only at run time).
RUN mkdir -p /sub /harness
WORKDIR /tmp
USER 65534:65534
