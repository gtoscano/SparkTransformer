### **1️⃣ When would you prefer an encoder model vs a decoder model?**

* **Encoder (e.g., BERT)** — use it when the task requires *understanding* or *classification* of existing text: sentiment analysis, NER, QA, embeddings for retrieval, etc.

  * In your run, the encoder quickly reached **~84% accuracy** on IMDB with 13 s of training — ideal for efficient supervised learning.
* **Decoder (e.g., GPT-2)** — use it when the task requires *generation*: open-ended responses, summarization, creative text, code, translation.

  * GPT-2 produced coherent 100-token completions but with **0.33 s latency per output** — much slower but necessary for text generation.

🧩 *Rule of thumb:* encoder = understanding; decoder = writing.

---

### **2️⃣ How does context length limit downstream tasks (long docs, code)?**

* **BERT:** limited to **512 tokens** — long documents are truncated, so you lose global context unless you use chunking or Longformer/BigBird variants.
* **GPT-2:** handles **1024 tokens**, allowing longer reasoning or storytelling but still constrained (modern GPT-4 models handle 128 k+).
* When documents or source code exceed the context window, both models “forget” earlier tokens — leading to incomplete understanding or incoherent generation.

🧠 *Practical fix:* use chunked inference, retrieval-augmented pipelines, or models with sparse or extended attention.

---

### **3️⃣ What trade-offs do sampling controls (temperature, top-k, top-p) introduce?**

| Control                | Effect                                                      | Trade-off                                              |
| ---------------------- | ----------------------------------------------------------- | ------------------------------------------------------ |
| **Temperature**        | Scales randomness; lower → deterministic, higher → creative | High = diverse but error-prone; low = factual but dull |
| **Top-k**              | Limits next-token candidates to k most likely               | Low k = focused but repetitive; high k = more variety  |
| **Top-p (nucleus)**    | Chooses smallest set whose probabilities sum to p           | Small p = safe; large p = diverse                      |
| **Repetition penalty** | Prevents loops (“the the the…”)                             | Too strong may distort grammar                         |

In your experiment:

* **Greedy** → factual, repetitive (“used to power a large number of...”)
* **Sampled** → more diverse and creative (“develop new therapeutics…”)
* **Prompt-controlled** → steered tone (“professional executive summary”).

⚖️ *You tune these knobs to balance creativity vs reliability.*

---

### **4️⃣ How does supervised fine-tuning (BERT) differ from prompt engineering (GPT-2)?**

| Aspect               | BERT (Encoder)                                           | GPT-2 (Decoder)                               |
| -------------------- | -------------------------------------------------------- | --------------------------------------------- |
| **Training signal**  | Supervised labels (loss = cross-entropy vs ground truth) | None — relies on pretraining + your prompt    |
| **Adaptation cost**  | Requires dataset & training loop (13 s in your test)     | Zero-shot or few-shot; just change prompt     |
| **Behavior control** | Precise & repeatable (e.g., always output class)         | Indirect; shaped by wording & decoding params |
| **Output type**      | Structured prediction (probabilities over classes)       | Free-form text generation                     |

🧩 *Fine-tuning changes weights; prompting changes context.*

---

### **5️⃣ For the same hardware, where is the time spent: supervised training vs autoregressive decoding?**

* **BERT fine-tuning:** ~**13 s total training** for 2000 samples, < 0.5 s eval — very efficient because it processes sequences **in parallel**.
* **GPT-2 generation:** **~326 ms** to produce 100 tokens — slow because decoding is **sequential**, each new token depends on all prior ones.
* Hence, **training encoders** is compute-intensive but short; **inference for decoders** is compute-intensive and slow per token.

⏱️ *Encoders parallelize across tokens → fast; decoders unroll across tokens → slow.*

---

✅ **Summary Table**

| Dimension | Encoder (BERT)                 | Decoder (GPT-2)                    |
| --------- | ------------------------------ | ---------------------------------- |
| Task      | Classification / understanding | Generation / completion            |
| Context   | 512 tokens                     | 1024 tokens                        |
| Speed     | Fast (3 ms inference)          | Slow (0.3 s per 100 tokens)        |
| Training  | Supervised fine-tuning         | Zero/few-shot prompting            |
| Control   | Deterministic                  | Temperature, top-k/p, prompt style |
| Output    | Label / embedding              | Text / code / story                |

---

In short:

> Use **encoders** when you need *answers*; use **decoders** when you need *language*.
