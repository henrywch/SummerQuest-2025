from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Paths
base_model_path = "/data-mnt/data/chwang/models/Qwen2.5-0.5B"
lora_adapter_path = "/root/SummerQuest-2025/submission/王宬皓/day-5/hw5_4/LLaMA-Factory/saves/qwen2.5-0.5b/lora/sft"
merged_model_path = "/data-mnt/data/chwang/models/Qwen2.5-0.5B-LoRA"

# Load base model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,  # Use bfloat16/float16 if supported
    device_map="auto",
    use_safetensors=True
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Load LoRA adapter and merge
model = PeftModel.from_pretrained(model, lora_adapter_path)
merged_model = model.merge_and_unload()  # Merge LoRA weights

# Save merged model and tokenizer
merged_model.save_pretrained(merged_model_path, safe_serialization=True)
tokenizer.save_pretrained(merged_model_path)