FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# Install System Tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl unzip ca-certificates bzip2 build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Install Deno
RUN curl -fsSL https://deno.land/x/install/install.sh | sh
ENV PATH="/root/.deno/bin:$PATH"

# Install Miniconda
RUN curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash miniconda.sh -b -p /opt/conda && rm miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

WORKDIR /app

# Accept tos
RUN /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# install mamba
RUN conda install -n base -c conda-forge mamba -y

# Create Whisper env
COPY env_whisper.yml .
RUN mamba env create -f env_whisper.yml && mamba clean -afy

# Create Facenet env 
COPY env_facenet.yml .
RUN mamba env create -f env_facenet.yml && mamba clean -afy

# Create Orchestrator env
COPY env_orchestrator.yml .
RUN mamba env create -f env_orchestrator.yml && mamba clean -afy

ENV PATH /opt/conda/envs/env_orchestrator/bin:$PATH

COPY . .