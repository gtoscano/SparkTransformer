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
teacher_id = "Qwen/Qwen2.5-7B"
student_id = "Qwen/Qwen2.5-1.5B"

# ----------------------------
# Load teacher and student
# ----------------------------
teacher = AutoModelForCausalLM.from_pretrained(teacher_id).eval().cuda()
student = AutoModelForCausalLM.from_pretrained(student_id).train().cuda()
tokenizer = AutoTokenizer.from_pretrained(teacher_id)

# ----------------------------
# Helper: Align vocab dimensions
# ----------------------------
def align_logits(teacher_logits, student_logits):
    """
    Pads logits along the last dimension (vocab size)
    so teacher and student have matching shapes.
    """

    t_vocab = teacher_logits.size(-1)
    s_vocab = student_logits.size(-1)

    if t_vocab == s_vocab:
        return teacher_logits, student_logits

    if s_vocab < t_vocab:
        # Student vocab smaller → pad student
        pad = t_vocab - s_vocab
        student_logits = F.pad(student_logits, (0, pad))
    else:
        # Teacher vocab smaller → pad teacher
        pad = s_vocab - t_vocab
        teacher_logits = F.pad(teacher_logits, (0, pad))

    return teacher_logits, student_logits

# ----------------------------
# Distillation Loss
# ----------------------------
def kd_loss(student_logits, teacher_logits, T=2.0):
    teacher_logits, student_logits = align_logits(teacher_logits, student_logits)

    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)

    return F.kl_div(s, t, reduction="batchmean") * (T * T)

# ----------------------------
# Forward pass
# ----------------------------
prompt = "Explain why quantization improves efficiency."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Teacher forward (no grad)
with torch.no_grad():
    teacher_logits = teacher(**inputs).logits

student_logits = student(**inputs).logits

# ----------------------------
# Compute KD Loss
# ----------------------------
loss = alpha * kd_loss(student_logits, teacher_logits, T=T)

# ----------------------------
# Print Info
# ----------------------------
print("Teacher logits:", teacher_logits.shape)
print("Student logits:", student_logits.shape)
print("KD loss:", loss.item())


# ----------------------------
# Optimizer
# ----------------------------

optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5)

# ----------------------------
# Training Loop (10 steps)
# ----------------------------
print("\n===== Training for 10 KD steps =====")

for step in range(1, steps + 1):
    optimizer.zero_grad()

    student_logits = student(**inputs).logits
    loss = alpha * kd_loss(student_logits, teacher_logits, T=T)

    loss.backward()
    optimizer.step()

    print(f"Step {step:02d} | KD Loss: {loss.item():.4f}")

print("\nTraining finished.")

