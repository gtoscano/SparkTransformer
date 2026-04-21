import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------
# Hyperparameters
# ----------------------------
T = 2.0          # temperature
alpha = 0.8      # weight on distillation loss
lr = 1e-6        # small LR for stability
steps = 10
max_grad_norm = 1.0

teacher_id = "meta-llama/Llama-3.2-3B-Instruct"
student_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ----------------------------
# Load teacher (FP16) and student (FP32)
# ----------------------------
teacher = AutoModelForCausalLM.from_pretrained(
    teacher_id,
    dtype=torch.float16,          # HF now prefers `dtype`
    device_map="cuda"
).eval()

# Student in full float32, then moved to GPU
student = AutoModelForCausalLM.from_pretrained(
    student_id,
    dtype=torch.float32
).to("cuda").train()

teacher_tok = AutoTokenizer.from_pretrained(teacher_id)
student_tok = AutoTokenizer.from_pretrained(student_id)

# ----------------------------
# Helper: Align vocab dimensions (last-token logits)
# ----------------------------
def align_logits(teacher_logits, student_logits):
    """
    Align teacher and student vocab dimensions by padding
    along the last dimension if needed.
    teacher_logits, student_logits: [batch, vocab]
    """
    t_vocab = teacher_logits.size(-1)
    s_vocab = student_logits.size(-1)

    if t_vocab == s_vocab:
        return teacher_logits, student_logits

    if s_vocab < t_vocab:
        pad = t_vocab - s_vocab
        student_logits = F.pad(student_logits, (0, pad))
    else:
        pad = s_vocab - t_vocab
        teacher_logits = F.pad(teacher_logits, (0, pad))

    return teacher_logits, student_logits

# ----------------------------
# Distillation Loss (in float32 for stability)
# ----------------------------
def kd_loss(student_logits, teacher_logits, T=2.0):
    # Align vocab sizes
    teacher_logits, student_logits = align_logits(teacher_logits, student_logits)

    # Cast to float32 for numerically stable softmax / log_softmax
    teacher_logits = teacher_logits.float()
    student_logits = student_logits.float()

    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)

    return F.kl_div(s, t, reduction="batchmean") * (T * T)

# ----------------------------
# Prepare inputs
# ----------------------------
prompt = "Explain why quantization improves efficiency."

# Use each model's own tokenizer for its input
teacher_inputs = teacher_tok(prompt, return_tensors="pt").to("cuda")
student_inputs = student_tok(prompt, return_tensors="pt").to("cuda")

# Teacher logits (no grad)
with torch.no_grad():
    teacher_full = teacher(**teacher_inputs).logits  # [B, T_teacher, V_teacher]
teacher_last = teacher_full[:, -1, :]                # [B, V_teacher]

# ----------------------------
# Optimizer
# ----------------------------
optimizer = torch.optim.AdamW(student.parameters(), lr=lr)

# ----------------------------
# Training Loop (10 steps)
# ----------------------------
print("\n===== Training for 10 KD steps (LLaMA 3.2 → TinyLlama, last-token KD) =====")

for step in range(1, steps + 1):
    optimizer.zero_grad()

    student_full = student(**student_inputs).logits    # [B, T_student, V_student]
    student_last = student_full[:, -1, :]              # [B, V_student]

    loss = alpha * kd_loss(student_last, teacher_last, T=T)

    if not torch.isfinite(loss):
        print(f"Step {step:02d} | Loss is not finite (loss={loss.item()}). Stopping.")
        break

    loss.backward()

    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)

    optimizer.step()

    print(f"Step {step:02d} | KD Loss: {loss.item():.4f}")

print("\nTraining finished.")

