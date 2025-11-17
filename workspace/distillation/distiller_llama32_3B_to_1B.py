import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------
# Hyperparameters
# ----------------------------
T = 2.0          # temperature
alpha = 0.8      # weight on distillation loss
lr = 1e-5
steps = 10

# LLaMA-style teacher / student
teacher_id = "meta-llama/Llama-3.2-3B-Instruct"   # larger teacher
student_id = "meta-llama/Llama-3.2-1B-Instruct"   # smaller student

# ----------------------------
# Load teacher and student
# ----------------------------
teacher = AutoModelForCausalLM.from_pretrained(
    teacher_id,
    dtype=torch.float16,
    device_map="cuda"
).eval()

student = AutoModelForCausalLM.from_pretrained(
    student_id,
    dtype=torch.float16,
    device_map="cuda"
).train()

tokenizer = AutoTokenizer.from_pretrained(teacher_id)

# ----------------------------
# Distillation Loss (no vocab hack needed)
# ----------------------------
def kd_loss(student_logits, teacher_logits, T=2.0):
    # shapes: [batch, seq_len, vocab_size] – same for both models
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)

# ----------------------------
# Forward pass
# ----------------------------
prompt = "Explain why quantization improves efficiency."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    teacher_logits = teacher(**inputs).logits

student_logits = student(**inputs).logits

loss = alpha * kd_loss(student_logits, teacher_logits, T=T)

print("Teacher logits:", teacher_logits.shape)
print("Student  logits:", student_logits.shape)
print("Initial KD loss:", loss.item())

# ----------------------------
# Optimizer
# ----------------------------
optimizer = torch.optim.AdamW(student.parameters(), lr=lr)

# ----------------------------
# Training Loop (10 steps)
# ----------------------------
print("\n===== Training for 10 KD steps (LLaMA-based) =====")

for step in range(1, steps + 1):
    optimizer.zero_grad()

    student_logits = student(**inputs).logits
    loss = alpha * kd_loss(student_logits, teacher_logits, T=T)

    loss.backward()
    optimizer.step()

    print(f"Step {step:02d} | KD Loss: {loss.item():.4f}")

print("\nTraining finished.")

