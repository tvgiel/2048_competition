# Evaluation image for the 2048 competition (Python + Julia + C++17, CUDA/GPU).
#
#   docker build -t 2048-eval:latest .
#   docker build -t 2048-eval:latest --build-arg WITH_TENSORFLOW=1 .
#
# The harness runs this image with --network none, a read-only root filesystem,
# a non-root user, resource limits, and --gpus to expose the GPU (see harness.py).
# Everything participants may import must be installed HERE, at build time: the
# sandbox has no network, so nothing can be downloaded during evaluation.
#
# GPU notes:
#   * The CUDA toolkit (nvcc + libs) comes from the CUDA -devel base below. The
#     host driver (libcuda.so) and device nodes are injected at run time by
#     `docker run --gpus ...` via the NVIDIA Container Toolkit, so the HOST needs
#     the NVIDIA driver (>= 550 for CUDA 12.4) and that toolkit installed.
#   * C++  : the harness compiles .cu agents with `nvcc -arch=sm_89`
#            (RTX 4000 Ada = compute capability 8.9).
#   * Py   : torch is the CUDA (cu124) wheel, NOT the CPU build.
#   * Julia: CUDA.jl is added and its runtime artifacts are baked at build time,
#            because there is no network at evaluation to download them lazily.
#
# Pin for reproducibility: use a fixed base digest (docker pull the tag, then
# FROM nvidia/cuda@sha256:...) and freeze the Julia Manifest (see bottom).
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
ENTRYPOINT []

ARG JULIA_VERSION=1.10.5
ARG WITH_TENSORFLOW=0
ENV DEBIAN_FRONTEND=noninteractive

# ---- System packages + Python 3.12 (Ubuntu 22.04 ships 3.10, so deadsnakes) --
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         software-properties-common curl ca-certificates g++ \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
         python3.12 python3.12-dev python3.12-venv \
    && curl -fsSL https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# ---- Julia ------------------------------------------------------------------
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

# ---- Python packages (CUDA torch; extend as the jury decides) ---------------
RUN python -m pip install --no-cache-dir \
        numpy==2.0.2 scipy==1.14.1 pandas==2.2.3 scikit-learn==1.5.2 numba==0.60.0 \
    && python -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 \
    && if [ "$WITH_TENSORFLOW" = "1" ]; then python -m pip install --no-cache-dir tensorflow==2.18.0; fi

# ---- Julia packages, precompiled + CUDA artifacts baked at build time -------
# JSON3 is required by shim.jl. Baking CUDA.jl's artifacts here means they are
# present under the (read-only) depot at eval time, when there is no network.
ENV JULIA_DEPOT_PATH=/opt/julia_depot
RUN mkdir -p /opt/julia_env \
    && julia --project=/opt/julia_env -e 'using Pkg; \
         Pkg.add(["JSON3", "Flux", "StatsBase", "Distributions", "CUDA"]); \
         using CUDA; try; CUDA.precompile_runtime(); catch e; @warn "CUDA runtime precompile skipped (no GPU at build?)" e; end; \
         Pkg.precompile()' \
    && chmod -R a+rX /opt/julia_env /opt/julia_depot

# To freeze exact Julia versions after the first build:
#   docker run --rm 2048-eval:latest cat /opt/julia_env/Manifest.toml > Manifest.toml
# then add before the RUN above:  COPY Manifest.toml /opt/julia_env/Manifest.toml

# Mount points used by the harness (rootfs is read-only at run time).
RUN mkdir -p /sub /harness
WORKDIR /tmp
USER 65534:65534
