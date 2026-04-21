# 🧠 Knowledge Distillation Examples

### LLaMA → LLaMA • LLaMA → TinyLlama • Qwen → Qwen • Cross-Family Distillation

This repository contains **clean, minimal examples** of several kinds of **LLM Knowledge Distillation (KD)**:

* Logit-level KD (same-family models)
* Cross-vocabulary KD with manual logit alignment
* Teacher-generated text KD (works across *any* model families — the most general and robust method)

These scripts are designed for **teaching** and **experimentation**, and each one demonstrates a different form of distillation.

---

## 🌟 What Is Knowledge Distillation?

**Knowledge Distillation (KD)** is a technique where a **large teacher model** trains a **smaller student model**, transferring knowledge so the student imitates the teacher.

There are three main forms of distillation:

### 1. **Logit-level KD**

Teacher and student produce logits of the same dimensionality.
We compute KL divergence between distributions:

[
KL(p_{\text{teacher}} \parallel p_{\text{student}})
]

This works best when **teacher and student share the same tokenizer**, vocabulary size, and architecture family.

---

### 2. **Cross-family logit KD (manual logit alignment)**

When teacher and student have **different vocabularies**, we must:

* Use separate tokenizers
* Compare only the final token
* Align vocab sizes by padding logits

This allows KD between different architectures, but requires careful handling.

---

### 3. **Text-based KD (teacher-generated text)**

This is the **most general**, **most stable**, and **easiest** form of distillation.

* Teacher model generates an explanation or answer
* Student model trains on this text using *standard cross-entropy*
* Works even if teacher and student:

  * use different tokenizers
  * have different vocab sizes
  * have different architectures
  * have different languages!

This method is widely used in practice because it bypasses vocab alignment issues.

> **If you want the most reliable, architecture-agnostic distillation method → use text KD.**

---

# 📁 Repository Structure

```
distiller_llama32_3B_to_1B.py
distiller_llama32_3B_to_tinyllama_logits.py
distiller_llama32_3B_to_tinyllama_text_kd.py
distiller_qwen_7b_to_1_5b.py
README.md
```

Below is a description of each script.

---

## 🔹 `distiller_llama32_3B_to_1B.py`

**Teacher:** LLaMA 3.2-3B
**Student:** LLaMA 3.2-1B
**KD type:** *Logit-level distillation (KL)*
**Tokenizer compatibility:** ✔ Same tokenizer → no alignment needed

This is the **cleanest and simplest** KD example:
Teacher and student share architecture, vocab, tokenizer → KL works out-of-the-box.

Use this example to learn the basics of **probability distribution matching**.

---

## 🔹 `distiller_llama32_3B_to_tinyllama_logits.py`

**Teacher:** LLaMA 3.2-3B
**Student:** TinyLlama 1.1B
**KD type:** *Cross-family logit KD (requires manual alignment)*
**Tokenizer compatibility:** ❌ Different tokenizers
**Vocab sizes:** ❌ Different sizes (must pad logits)

This example demonstrates:

* Why logits cannot be compared directly across model families
* How to align vocab sizes using padding
* Why we must tokenize teacher and student **separately**
* How to perform KD on **last-token logits**

Useful for teaching the complications of cross-family KD.

---

## 🔹 `distiller_llama32_3B_to_tinyllama_text_kd.py`

**Teacher:** LLaMA 3.2-3B
**Student:** TinyLlama 1.1B
**KD type:** ***Text-based distillation (teacher-generated text)***
**Tokenizer compatibility:** ✔ Works with ANY tokenizer
**Vocab sizes:** ✔ Irrelevant
**Stability:** ✔ Very stable (no FP16 issues)

This script shows the **most general and robust distillation method**:

1. Teacher model generates an answer/explanation
2. Student is trained on the teacher-generated text
3. Standard cross-entropy loss (no KL)
4. Works across ANY model combinations

> **This is the recommended method for cross-family distillation.**
> Perfect for showing that “models of different nature can teach each other.”

---

## 🔹 `distiller_qwen_7b_to_1_5b.py`

**Teacher:** Qwen 2.5-7B
**Student:** Qwen 2.5-1.5B
**KD type:** *Logit-level distillation (KL)*
**Tokenizer compatibility:** ✔ Same family
**Vocab sizes:** ❌ Slightly different → script includes logit alignment

This is a real-world example where even models from the *same family* sometimes differ in vocabulary and require minor alignment.

Great for demonstrating:

* How vocabulary mismatches appear
* How to pad logits safely
* How KD performs across multiple model sizes in a real family

---

# 🏁 Summary

| Script                               | Teacher → Student | KD Type            | General?           | Notes                           |
| ------------------------------------ | ----------------- | ------------------ | ------------------ | ------------------------------- |
| `llama32_3B_to_1B.py`                | LLaMA → LLaMA     | Logit KL           | ✖ same family only | Cleanest KL example             |
| `llama32_3B_to_tinyllama_logits.py`  | LLaMA → TinyLlama | Logit KL (aligned) | ✖ limited          | Requires vocab alignment        |
| `llama32_3B_to_tinyllama_text_kd.py` | LLaMA → TinyLlama | **Text KD**        | **✔ MOST GENERAL** | Easiest + most robust           |
| `qwen_7b_to_1_5b.py`                 | Qwen → Qwen       | Logit KL + align   | ✖ same family      | Shows real-world vocab mismatch |

---

# 💡 Final Recommendation

If your goal is to **teach how different models can teach each other**,
the best demonstration is:

👉 **`distiller_llama32_3B_to_tinyllama_text_kd.py`**

Because:

* It works across any architectures
* It is stable in FP16/FP32
* It avoids logit shape mismatch headaches
* It mimics real-world “self-training” and “synthetic dataset” pipelines
* Students see actual model imitation before/after distillation
