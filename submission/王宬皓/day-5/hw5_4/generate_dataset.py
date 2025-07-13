# generate_data.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import json
import os
import time
from tqdm import tqdm

# 配置参数
MODEL_PATH = "/data-mnt/data/chwang/models/Qwen2.5-Math-7B"
DATA_PATH = "data/gsm8k"
OUTPUT_DIR = "dataset"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gsm8k_cot.jsonl")

# 修复提示模板 - 转义花括号
PROMPT_TEMPLATE = (
    "You are a math expert. Please solve the question step by step and put the final answer in \\boxed{{}}.\n\nQuestion: {question}"
)

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载数据集
try:
    print(f"Loading dataset from {DATA_PATH}...")
    dataset = load_dataset(DATA_PATH, "main", split="train")
    questions = [ex["question"] for ex in dataset][:1000]  # 只处理前10个问题测试
    print(f"Loaded {len(questions)} questions from GSM8K dataset")
    
    # 打印前2个问题以验证数据格式
    print("\nSample questions:")
    for i, q in enumerate(questions[:2]):
        print(f"{i+1}. {q}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# 初始化模型和tokenizer
try:
    print(f"\nLoading model from {MODEL_PATH}...")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).eval()
    
    print("Model loaded successfully")
    
    # 打印模型信息
    print(f"Model device: {model.device}")
    print(f"Model dtype: {model.dtype}")
    
    # 添加填充token设置（如果未设置）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")
        
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# 生成函数 - 单样本处理
def generate_response(prompt):
    """为单个提示生成响应"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return response.strip()

# 生成函数 - 批量处理
def generate_batch_responses(prompts, batch_size=4):
    """批量生成响应，更高效"""
    responses = []
    
    # 使用进度条
    pbar = tqdm(total=len(prompts), desc="Generating responses")
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        try:
            # 批量编码
            inputs = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=512,  # 限制输入长度
                return_token_type_ids=False
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # 解码响应
            for j in range(len(batch_prompts)):
                # 获取实际生成的token（去除输入部分）
                input_length = inputs["input_ids"][j].shape[0]
                output_tokens = outputs[j][input_length:]
                
                response = tokenizer.decode(
                    output_tokens, 
                    skip_special_tokens=True
                )
                responses.append(response.strip())
                
            pbar.update(len(batch_prompts))
            
        except Exception as e:
            print(f"\nError processing batch starting at index {i}: {e}")
            # 为当前批次中的每个提示生成错误响应
            for _ in batch_prompts:
                responses.append(f"ERROR: {str(e)}")
            pbar.update(len(batch_prompts))
    
    pbar.close()
    return responses

# 批量生成回答
print(f"\nGenerating COT for {len(questions)} questions...")
start_time = time.time()

# 创建所有提示 - 使用修复后的模板
prompts = []
for q in questions:
    try:
        # 安全格式化 - 处理可能的问题字符
        prompt = PROMPT_TEMPLATE.format(question=q.replace('{', '{{').replace('}', '}}'))
        prompts.append(prompt)
    except Exception as e:
        print(f"Error formatting prompt for question: {e}")
        # 使用原始问题作为回退
        prompts.append(f"You are a math expert. Please solve the question step by step and put the final answer in \\boxed{{}}.\n\nQuestion: {q}")

# 选择生成方式
use_batch_processing = True
batch_size = 4  # 根据GPU内存调整

with open(OUTPUT_FILE, "w") as f:
    if use_batch_processing:
        print(f"Using batch processing with batch size {batch_size}...")
        responses = generate_batch_responses(prompts, batch_size)
        
        # 保存结果
        for i, (question, response) in enumerate(zip(questions, responses)):
            result = {
                "instruction": "You are a math expert. Please solve the question step by step.",
                "input": question,
                "output": response
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"✅ Saved response for question {i+1}/{len(questions)}")
    else:
        print("Using single sample processing...")
        # 单样本处理
        for i, (prompt, question) in enumerate(tqdm(zip(prompts, questions), desc="Processing questions")):
            try:
                response = generate_response(prompt)
                result = {
                    "instruction": "You are a math expert. Please solve the question step by step.",
                    "input": question,
                    "output": response
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"❌ Error generating response for question {i+1}: {e}")
                # 保存错误信息
                result = {
                    "instruction": "Solve the math problem step by step.",
                    "input": question,
                    "output": f"ERROR: {str(e)}"
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

end_time = time.time()
print(f"\nSaved COT responses to {OUTPUT_FILE}")
print(f"Total time: {end_time - start_time:.2f} seconds")
print(f"Average time per question: {(end_time - start_time)/len(questions):.2f} seconds")
print("Data generation completed!")

# 打印样本输出
try:
    print("\nSample output:")
    with open(OUTPUT_FILE, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:min(3, len(lines))]):
            data = json.loads(line)
            print(f"Sample {i+1}:")
            print(f"Input: {data['input'][:100]}...")
            print(f"Output: {data['output'][:200]}...\n")
except Exception as e:
    print(f"Error reading output file: {e}")