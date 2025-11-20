docker run --gpus all -it --rm -p 8000:8000 -p 8001:8001   -p 11434:11434 -v "$(pwd)/model_repository:/models" nemo-guardrails-triton:latest

