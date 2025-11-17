import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------
# Hyperparameters
# ----------------------------
lr = 1e-5
steps = 10
max_new_tokens = 64

teacher_id = "meta-llama/Llama-3.2-3B-Instruct"
student_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

device = "cuda"

# ----------------------------
# Load teacher (FP16) and student (FP32 for stability)
# ----------------------------
teacher = AutoModelForCausalLM.from_pretrained(
    teacher_id,
    dtype=torch.float16,
    device_map=device,
).eval()

student = AutoModelForCausalLM.from_pretrained(
    student_id,
    dtype=torch.float32,
).to(device).train()

teacher_tok = AutoTokenizer.from_pretrained(teacher_id)
student_tok = AutoTokenizer.from_pretrained(student_id)

# ----------------------------
# Step 1: Teacher generates a target explanation
# ----------------------------
prompt = "Explain why quantization improves efficiency."

teacher_inputs = teacher_tok(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    gen_ids = teacher.generate(
        **teacher_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

teacher_text = teacher_tok.decode(gen_ids[0], skip_special_tokens=True)
print("=== Teacher-generated text ===")
print(teacher_text)
print("==============================\n")

# ----------------------------
# Step 2: Prepare student training example
# ----------------------------
# For simplicity: student learns to model the full teacher text as a sequence.
# (You could also keep the prompt separate and only train on the continuation.)
student_inputs = student_tok(
    teacher_text,
    return_tensors="pt",
).to(device)

input_ids = student_inputs["input_ids"]           # [1, L]
attention_mask = student_inputs["attention_mask"]

# Standard causal LM setup: labels = input_ids
labels = input_ids.clone()

# ----------------------------
# Optimizer
# ----------------------------
optimizer = torch.optim.AdamW(student.parameters(), lr=lr)

# ----------------------------
# Training loop (10 steps)
# ----------------------------
print("===== Training TinyLlama on teacher text (LLaMA 3.2 → TinyLlama) =====")

for step in range(1, steps + 1):
    optimizer.zero_grad()

    outputs = student(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,    # Hugging Face computes cross-entropy internally
    )
    loss = outputs.loss  # scalar

    if not torch.isfinite(loss):
        print(f"Step {step:02d} | Loss is not finite (loss={loss.item()}). Stopping.")
        break

    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    optimizer.step()

    print(f"Step {step:02d} | Student loss: {loss.item():.4f}")

print("\nTraining finished.\n")

# ----------------------------
# Step 3: Show student imitation after training
# ----------------------------
student_prompt = "Explain why quantization improves efficiency."
student_inputs_gen = student_tok(student_prompt, return_tensors="pt").to(device)

with torch.no_grad():
    student_gen_ids = student.generate(
        **student_inputs_gen,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

student_text = student_tok.decode(student_gen_ids[0], skip_special_tokens=True)

print("=== Student-generated text AFTER distillation ===")
print(student_text)
print("=================================================")

