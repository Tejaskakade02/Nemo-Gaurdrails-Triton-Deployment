# 🚀 NeMo Guardrails + NVIDIA Triton + Ollama

A complete production-grade setup for running **NeMo Guardrails** inside **NVIDIA Triton Inference Server**, using **Ollama** as the backend LLM engine. This repository lets you run:

* ✔ Llama 3 / Mistral / any Ollama model inside Triton
* ✔ NeMo Guardrails (safety, policy, intent, conversation flows)
* ✔ A fully containerized deployment (Docker)
* ✔ Triton Python backend with custom `model.py`
* ✔ REST/gRPC inference via Triton API

This project is ideal for **production chatbots**, **enterprise safety layers**, and **secure inference pipelines**.

---

# 📦 Features

### ✅ Triton-hosted Guardrails Engine

All safety logic runs inside Triton using the Python backend.

### ✅ Ollama LLM Integration

Uses Ollama locally to run models efficiently (GPU or CPU).

### ✅ Supports NeMo Guardrails v0.16.x

Fully tested with Colang 1.0 syntax and YAML config.

### ✅ Works with Any Chat LLM

Examples:

* llama3:8b-instruct
* mistral:7b
* phi3
* custom GGUF models

### ✅ Low VRAM Friendly

Optimized for laptops with RTX 4050 / 4–6 GB VRAM.

---

# 📁 Repository Structure

```
model_repository/
│
└── guardrails/
    ├── 1/
    │   ├── model.py
    │   └── rails/
    │       ├── config.yml
    │       └── rails.co
    │
    └── config.pbtxt

Dockerfile
README.md  ← YOU ARE HERE
scripts/
└── test_guardrails.py
```

---

# 🐳 1. Build the Docker Image

```bash
docker build -t nemo-guardrails-triton .
```

Make sure Ollama is installed inside the image using:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

The image automatically pulls your model on first run.

---

# 🖥️ 2. Start Triton + Ollama

Run the container:

```bash
docker run --gpus all -it --rm \
  -p 8000:8000 -p 8001:8001 -p 11434:11434 \
  -v "$(pwd)/model_repository:/models" \
  nemo-guardrails-triton
```

If the LLM model is not downloaded, Ollama will automatically fetch it.

---

# 📘 3. Guardrails Configuration

### `config.yml`

Defines the LLM backend, temperatures, intents, and bot policies.

Example:

```yaml
models:
  main:
    type: main
    engine: ollama
    model: llama3:8b-instruct-q5_K_M
    parameters:
      base_url: "http://127.0.0.1:11434"
      temperature: 0.3

rails:
  input:
    user:
      - intent: user_input
  output:
    bot:
      - type: bot_response

colang_files:
  - rails.co
```

---

### `rails.co`

You can define intents, safety, and flows:

```
define user_input
  user says anything
end

define bot_response
  bot says something
end
```

This is Colang v1.0 syntax.

---

# 🧠 4. Triton Model (`model.py`)

This Python backend file integrates Triton ↔ NeMo Guardrails ↔ Ollama.

> This version is compatible with NeMo Guardrails **0.16.x**.

Highlights:

* Loads Guardrails config
* Forwards user messages to LLM
* Returns bot message
* Handles string + dict outputs safely

The full working file is included in `model.py`.

---

# 🧪 5. Testing the Guardrails Engine

Use the included test script:

```bash
python3 scripts/test_guardrails.py
```

You should see:

```
===============================
  Testing LLM + Guardrails
===============================

User: Hello!
Bot: Hi! How can I assist you today?
```

---

# 🔧 Troubleshooting

### ❗ Error: model not found (404)

Run:

```bash
ollama pull llama3:8b-instruct-q5_K_M
```

---

### ❗ KeyError: 'messages'

Means Guardrails v0.16 returned a **string**.
Use the latest `model.py` from this repo.

---

### ❗ AttributeError: 'str' has no attribute 'get'

Same issue — incorrect parsing.
Your `model.py` must normalize strings.

---

### ❗ GPU low VRAM mode

Ollama automatically enters low VRAM mode on 4–6GB GPUs.

This is normal.

---

# ✨ Performance Tips

* Use quantized models: `q4_K_M`, `q5_K_M`.
* Use `num_gpu=20` for RTX 4050.
* Reduce context length to 4096.
* Use smaller batch sizes.

---

# 🛡 Security

NeMo Guardrails provides:

* Content moderation
* Intent detection
* Custom safety rules
* Policy enforcement
* Conversation flow control

Suitable for enterprise deployments.

---

# 🤝 Contributing

Pull requests are welcome! Feel free to open:

* Issues
* Bug reports
* Feature requests

---

# 📄 License

This project is licensed under the MIT License.

---

# ⭐ Support

If this helped you, please star ⭐ the repository!

# Output
![Model Output](assets/nemo_up.png)

![Model Output](assets/test_nemo.png)

