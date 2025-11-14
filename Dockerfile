
# Use Triton Inference Server with Python backend
FROM nvcr.io/nvidia/tritonserver:24.09-py3

# Work directory
WORKDIR /workspace

# Create models directory
RUN mkdir -p /models

# Copy Triton model repository
COPY model_repository /models

# Install OS dependencies
RUN apt-get update -qq && \
    apt-get install -y -qq curl wget git && \
    apt-get clean
# Install Ollama inside the container
RUN curl -fsSL https://ollama.com/install.sh | sh

# ---- FIXED DEPENDENCIES (100% COMPATIBLE WITH NEMO 0.17) ----
RUN pip install --no-cache-dir \
    nemoguardrails==0.16.0 \
    langchain \
    langchain-core \
    langchain-community \
    langchain-ollama \
    nest-asyncio \
    gradio


# Expose Triton ports
EXPOSE 8000 8001 8002


# Start Ollama first, then Triton
CMD ollama serve & \
    sleep 3 && \
    tritonserver --model-repository=/models --log-verbose=1


