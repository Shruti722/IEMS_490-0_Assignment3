# Dockerfile for LLM Assignment 3 – RLHF with PPO, GRPO, and DPO

FROM python:3.10-slim

# Basic Python settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Workdir inside the container
WORKDIR /workspace

# Install minimal OS deps (optional but useful)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch \
    transformers \
    datasets \
    accelerate

# Copy the repo into the container
COPY . .

# Default command: drop into a shell so the grader can run any script
CMD ["bash"]
