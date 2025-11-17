# ⚡️ Using TensorRT-LLM with SparkTransformer (CUDA 13, `trtllm-serve`)

This guide assumes the `trtllm` service is already defined in your `docker-compose.yml` and explains how to authenticate with NVIDIA NGC, start the server, and use TensorRT-LLM inside SparkTransformer.

---

# 🔐 0. **Before You Begin — Authenticate to NVIDIA NGC**

To pull the `nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc2` container,
you **must authenticate** with NVIDIA's NGC registry.

### Step 1 — Create or log into your NGC account

Go to:
**[https://ngc.nvidia.com](https://ngc.nvidia.com)**

If you belong to an institution, log in with your organizational email.

---

### Step 2 — Generate an NGC API Key

1. Visit the API key page:
   **[https://org.ngc.nvidia.com/setup/api-keys](https://org.ngc.nvidia.com/setup/api-keys)**
2. Click **“Generate API Key”**.
3. Copy the key (you will not be able to view it again).

---

### Step 3 — Log in to the `nvcr.io` Docker registry

In your terminal:

```bash
docker login nvcr.io
```

When prompted:

* **Username:** `$oauthtoken`
* **Password:** *<your NGC API key>*

Example:

```
Username: $oauthtoken
Password: <paste your NGC API key>
Login Succeeded
```

### ✔ You are now authenticated to pull TensorRT-LLM

If you skip this step, Docker will return:

```
Access Denied
```

or:

```
unauthorized: authentication required
```

---

# 🧱 1. Start the TensorRT-LLM Service (Docker Compose)

TensorRT-LLM is already included in `docker-compose.yml`:

```yaml
trtllm:
  image: nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc2
  gpus: all
  ipc: host
  shm_size: "16g"
  ulimits:
    memlock: -1
    stack: 67108864

  ports:
    - "8000:8000"

  env_file:
    - .env

  environment:
    - HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}

  command: >
    trtllm-serve
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    --host 0.0.0.0
    --port 8000
    --trust_remote_code
```

Start the service:

```bash
docker compose up -d trtllm
```

Check logs:

```bash
docker logs -f sparktransformer-trtllm-1
```

You should see:

```
[TensorRT-LLM] TensorRT LLM version: 1.2.0rc2
Downloading TinyLlama/TinyLlama-1.1B-Chat-v1.0...
Serving on 0.0.0.0:8000
```

---

# 💬 2. Query TensorRT-LLM from the Host

Use its OpenAI-compatible API:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Say hello world!."}
        ],
        "max_tokens": 32
      }'
```

---

# 🧠 3. Query TensorRT-LLM from the Trainer Container

Enter the trainer:

```bash
docker compose exec trainer bash
```

Then:

```python
python - << 'PY'
import requests, json
url = "http://trtllm:8000/v1/chat/completions"

payload = {
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello from the trainer container!"}
    ],
    "max_tokens": 32
}

print(json.dumps(requests.post(url, json=payload).json(), indent=2))
PY
```

Since both services are on the same Docker network,
`http://trtllm:8000/...` resolves automatically.

---

# 🔁 4. Switching the Served Model

To serve a different Hugging Face model:

```yaml
command: >
  trtllm-serve
  "meta-llama/Meta-Llama-3-8B-Instruct"
  --host 0.0.0.0
  --port 8000
  --trust_remote_code
```

Then restart:

```bash
docker compose up -d trtllm
```

---

# ⚡ 5. Why Use TensorRT-LLM?

You get:

* Extremely high **tokens/sec** performance
* FP16 / BF16 / FP8 kernels optimized for Hopper
* Multi-GPU execution
* OpenAI-API compatibility

SparkTransformer handles experimentation + fine-tuning.
TensorRT-LLM handles high-performance serving.

---

# ✔ Summary

To use TensorRT-LLM:

1. **Generate NGC API Key** → [https://org.ngc.nvidia.com/setup/api-keys](https://org.ngc.nvidia.com/setup/api-keys)
2. **Login to nvcr.io** using `$oauthtoken` + API key
3. `docker compose up -d trtllm`
4. Query via OpenAI API (`localhost:8000`)
5. Switch models easily by editing the `command:` field

You now have a complete **train → optimize → serve** pipeline:
**SparkTransformer → TensorRT-LLM → DGX Spark GPUs**.

