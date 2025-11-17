from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

# 1. Base model name (must match training)
model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model_name = "meta-llama/Llama-3.2-3B-Instruct"
# model_name = "Qwen/Qwen2.5-1.5B"
# model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# 2. LoRA/QLoRA adapter folder (must match save path in training)
lora_dir = "./qlora-out"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 3. 4-bit quantized base model (same as training → QLoRA style)
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("Loading 4-bit base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_cfg,
    device_map="auto",
)
base_model.config.pad_token_id = tokenizer.pad_token_id

print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, lora_dir)
model = model.eval()

# 4. Run inference
prompt = "Write a positive movie review about a space adventure."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

print("\n=== Model Output ===\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

