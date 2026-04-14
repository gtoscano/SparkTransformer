---
marp: true
theme: default
paginate: true
backgroundColor: #1a1a2e
color: #eaeaea
style: |
  section {
    font-family: 'Segoe UI', sans-serif;
  }
  h1 {
    color: #00d4ff;
  }
  h2 {
    color: #00d4ff;
  }
  table {
    font-size: 0.7em;
    margin: 0 auto;
  }
  th {
    background-color: #16213e;
    color: #00d4ff;
  }
  td {
    background-color: #0f3460;
  }
  code {
    background-color: #16213e;
    color: #e94560;
  }
  strong {
    color: #e94560;
  }
  a {
    color: #00d4ff;
  }
---

# Self-Hosted LLM Servers for a CS Department

Running open-source models on **2x NVIDIA RTX PRO 6000** (97GB each)

---

# Why Self-Host?

- **Privacy**: Student code never leaves your network
- **Cost**: No per-token API fees for an entire department
- **Control**: Choose models, set rate limits, monitor usage
- **Availability**: No API outages or rate limits
- **Education**: Students learn about LLM infrastructure

---

# The Hardware

**2x NVIDIA RTX PRO 6000 Blackwell** (97GB VRAM each)

- Blackwell architecture (SM120)
- 97GB GDDR7 per GPU
- Running **vLLM** with OpenAI-compatible API
- Docker Compose for easy deployment

---

# Understanding LLM Benchmarks

---

## MMLU-Pro (General Knowledge)

**Massive Multitask Language Understanding - Professional**

- 10-choice questions across **14 domains**: STEM, humanities, social sciences, medicine, law
- Harder than original MMLU — requires reasoning, not just recall
- Score of **85+** is GPT-4 class

*"Can the model answer exam questions across disciplines?"*

---

## GPQA Diamond (Graduate-Level Science)

**Graduate-Level Google-Proof Q&A**

- 448 expert-written questions in biology, physics, chemistry
- Questions that **PhD students** struggle with
- "Google-proof" — can't be solved by simple search

*"Can the model reason at a graduate research level?"*

---

## LiveCodeBench v6 (Coding)

**Real-world competitive programming problems**

- Problems published **after** model training cutoffs
- Tests code generation from problem descriptions
- Solutions must pass automated test cases
- Multiple languages (Python, C++, Java)

*"Can the model solve LeetCode-style problems correctly?"*

---

## SWE-bench Verified (Software Engineering)

**Real GitHub issues from popular open-source repos**

- Model must understand a codebase, locate the bug, write a fix
- Tests are run to verify the patch works
- Closest benchmark to **real-world software engineering**

*"Can the model fix real bugs in real codebases?"*

---

## AIME 2025 (Math Reasoning)

**American Invitational Mathematics Examination**

- Competition-level math problems
- Requires multi-step reasoning and creative problem solving
- Tests deep mathematical thinking, not just arithmetic

*"Can the model solve hard math competition problems?"*

---

# Model Comparison

---

## Benchmarked on Our Hardware

| Model | Params | Active | tok/s | Weights | KV Cache | MMLU-Pro | GPQA | LCB v6 | SWE | AIME |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-puzzle-88B | 88B | 5.1B | **178** | ~50GB | ~43GB | 79.2 | 75.3 | — | — | 40.8 |
| **Qwen3.5-35B-A3B** | 35B | 3B | **140** | ~13GB | **~80GB** | **85.3** | **84.2** | **74.6** | 69.2 | — |
| Qwen3-Coder-Next | 80B | 3B | 106 | ~20GB | ~73GB | 80.5 | 74.5 | 58.9 | **70.6** | **83.1** |
| Qwen3.5-122B-A10B | 122B | 10B | 79 | ~76GB | ~17GB | **86.7** | **86.6** | **78.9** | **72.0** | — |
| Devstral-Small-2-24B | 24B | 24B | 56 | ~24GB | ~69GB | — | — | — | 68.0 | — |

---

## Why MoE Models Win

**Mixture of Experts (MoE)** activates only a fraction of parameters per token

- **Qwen3.5-35B-A3B**: 35B total but only **3B active** per token
- Result: Fast inference + large knowledge base
- More KV cache = **more concurrent students**

| | Dense 24B | MoE 35B (3B active) |
|---|---|---|
| Speed | 56 tok/s | **140 tok/s** |
| Knowledge | 24B | 35B |
| Concurrent users | Medium | **High** |

---

# Our Recommended Setup

| GPU | Model | Port | Use Case |
|---|---|---|---|
| GPU 0 | Qwen3-Coder-Next-NVFP4 | 8355 | Aider / agentic coding |
| GPU 1 | Qwen3.5-35B-A3B-NVFP4 | 8356 | General QA / chat |

Both models: **3B active params, ~20GB weights, ~73-80GB KV cache**

High concurrency for many simultaneous students

---

# Agentic Coding Tools

---

## What is Agentic Coding?

Traditional autocomplete: *suggests the next line*

Agentic coding: **understands your entire codebase, plans changes, edits multiple files, runs tests, and iterates**

The model becomes a **coding partner**, not just a suggestion engine.

---

## Aider

**Open-source terminal AI pair programmer** (39K GitHub stars)

- Maps your **entire codebase** for context-aware edits
- Makes **git commits** automatically
- Works with **any OpenAI-compatible API** (our vLLM servers)

```bash
aider --openai-api-base http://localhost:8355/v1 \
      --openai-api-key unused \
      --model RedHatAI/Qwen3-Coder-Next-NVFP4
```

Best with: **Qwen3-Coder-Next** (SWE-bench 70.6, AIME 83.1)

---

## Mistral Vibe CLI

**Mistral's terminal coding agent** (released with Devstral 2)

- Explore and modify codebases via natural language
- **7x more cost-efficient** than Claude Sonnet on real-world tasks
- Powered by Devstral 2 (72.2% SWE-bench)

Works with self-hosted Devstral models via OpenAI-compatible API

---

## Claude Code

**Anthropic's agentic CLI tool**

- Understands codebase, executes tasks, handles git workflows
- Available as CLI + extensions for VS Code, Cursor, JetBrains
- Requires Anthropic API (not self-hostable)

The gold standard for agentic coding — **but requires paid API access**

---

## IDE-Based AI Tools

| Tool | Type | Self-Hosted? | Key Feature |
|---|---|---|---|
| **Cursor** | VS Code fork | Partial | Background agents in isolated VMs |
| **Continue.dev** | IDE extension | **Yes** | Fully open-source, any LLM |
| **GitHub Copilot** | IDE extension | No | Custom agents via .agent.md |
| **Cline** | IDE extension | **Yes** | Conservative review-first workflow |
| **OpenHands** | SDK | **Yes** | Model-agnostic agent framework |

**Continue.dev** and **Cline** work directly with our self-hosted vLLM servers

---

## Connecting to Our Servers

All tools that support **OpenAI-compatible APIs** work with our setup:

```
API Base URL:  http://<server>:8355/v1   (Coding)
               http://<server>:8356/v1   (General QA)
API Key:       unused
```

Compatible with: Aider, Continue.dev, Cline, Cursor, OpenHands, and any OpenAI SDK client

---

# Running Both Models

```bash
# GPU 0 — Agentic coding (port 8355)
docker compose -f docker-compose-vllm.yml up -d

# GPU 1 — General QA (port 8356)
docker compose -f docker-compose-vllm-qwen35.yml up -d
```

Test:
```bash
curl http://localhost:8355/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "RedHatAI/Qwen3-Coder-Next-NVFP4",
       "messages": [{"role": "user", "content": "What is 2+2?"}],
       "max_tokens": 100}'
```

---

# Summary

- **Self-hosted LLMs** give a CS department privacy, control, and zero API costs
- **MoE models** provide the best throughput for many concurrent users
- **Qwen3.5-35B-A3B** is the best all-rounder (85.3 MMLU, 74.6 LCB, 140 tok/s)
- **Qwen3-Coder-Next** excels at agentic coding (70.6 SWE-bench, 83.1 AIME)
- Tools like **Aider**, **Continue.dev**, and **Cline** connect directly to our servers
- Two 97GB GPUs serve an entire department simultaneously

---

# Questions?

**Server endpoints:**
- Coding: `http://<server>:8355/v1`
- General QA: `http://<server>:8356/v1`

All compose files and test scripts available in the `LLMServers/` directory
