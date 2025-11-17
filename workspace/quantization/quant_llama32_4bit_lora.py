from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch

# 1. Choose your Llama 3.2 model
model_id = "meta-llama/Llama-3.2-1B-Instruct"  # or 3B/11B etc.
#model_id = "meta-llama/Llama-3.2-3B-Instruct"  # or 3B/11B etc.


# 2. 4-bit quantization config (nf4 + fp16 compute)
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("Loading tokenizer and model...")
tok = AutoTokenizer.from_pretrained(model_id)
base = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb,
    device_map="auto",
)

# 3. LoRA config (attention projections)
cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(base, cfg)
tok.pad_token = tok.eos_token  # important for generate()

# 4. Helper function to sample with given temperature/top_p
def sample_answer(prompt: str, temperature: float = 0.7, top_p: float = 0.9, max_new_tokens: int = 80):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,             # <-- sampling ON
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    # 5. Simple example: Where is located the Catholic University of America WITH sampling
    base_prompt = "Where is The Catholic University of America located ?"
    print("=== Sampling demo: 'Where CUA is located' ===")
    for i in range(3):
        torch.manual_seed(i)  # change seed to show different outputs
        ans = sample_answer(base_prompt, temperature=0.7, top_p=0.9, max_new_tokens=20)
        print(f"\nRun {i+1}:")
        print(ans)

