# MLLMsent — Multimodal LLMs see sentiment
#
# Ships the research framework as a ready-to-use environment provisioned with uv
# (the same workflow used for local development).
#
# DeepSeek-VL2 inference additionally needs the non-PyPI package (run inside the
# container when required):
#   pip install git+https://github.com/deepseek-ai/DeepSeek-VL2.git
FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .

# Environment only — drop into a shell; nothing is executed automatically.
CMD ["bash"]
