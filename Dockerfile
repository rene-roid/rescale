FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
# ffmpeg: audio decoding for whisperx/demucs. cuDNN 9 + CUDA 12.8 match the torch 2.8 cu128 wheels in uv.lock.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/python UV_COMPILE_BYTECODE=1
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
COPY packages ./packages
RUN uv sync --frozen --no-dev
COPY config ./config
COPY frontend ./frontend
ENV RESCALE_LIBRARY_ROOT=/music RESCALE_API_HOST=0.0.0.0 RESCALE_API_PORT=8765
EXPOSE 8765
CMD ["uv", "run", "--frozen", "--no-dev", "rescale", "serve"]
