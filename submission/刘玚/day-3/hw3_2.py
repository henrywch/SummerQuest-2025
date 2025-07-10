import os
import time
import json
from typing import List, Dict
import vllm
from transformers import AutoTokenizer

# 初始化 vLLM 引擎
print("=== vLLM 引擎初始化 ===")
print("正在初始化 vLLM 引擎...")
print("注意: vLLM 初始化可能需要几分钟时间")

tokenizer = AutoTokenizer.from_pretrained("./tokenizer_with_special_tokens", trust_remote_code=True)

# vLLM 引擎配置
llm = vllm.LLM(
    model="/remote-home1/share/models/Qwen3-8B",
    gpu_memory_utilization=0.8, 
    trust_remote_code=True,
    enforce_eager=True,
    max_model_len=4096,
)

print("vLLM 引擎和分词器初始化完成！")

# 读取查询数据
with open('query_only.json', 'r', encoding='utf-8') as f:
    queries = json.load(f)

# 配置采样参数
sampling_params = vllm.SamplingParams(
    temperature=0.7,
    top_p=0.8,
    max_tokens=2048,
    stop=None,
)

# 定义工具列表 - 符合Qwen格式
tools = [
    {
        "type": "function",
        "function": {
            "name": "python",
            "description": "Execute Python code for debugging and analysis",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "editor",
            "description": "Edit and merge code by comparing original and modified versions",
            "parameters": {
                "type": "object",
                "properties": {
                    "original_code": {
                        "type": "string",
                        "description": "Original code before modification"
                    },
                    "modified_code": {
                        "type": "string",
                        "description": "Modified code after fixing"
                    }
                },
                "required": ["original_code", "modified_code"]
            }
        }
    }
]

def generate_prompt(query: str) -> str:
    """
    为单个查询生成prompt
    """
    system_content = """你是代码调试助手。请严格按照以下步骤回答：

第一步：在<think></think>标签内分析用户是否给出明确报错信息.
第二步：必须二选一使用特殊标记：
如果用户提到了"报错信息"或具体的错误类型，使用<|EDIT|>
如果用户只是说代码"有问题"、"有bug"但没有报错信息，使用<|AGENT|>

请严格遵守以下输出格式：
<|EDIT|>
修复说明{"name": "editor", "arguments": {"code": "修复后的完整代码"}}

<|AGENT|>
分析说明{"name": "python", "arguments": {"code": "用户提供的代码"}}

示例1（有报错信息用<|EDIT|>）：
用户："报错信息如下： ZeroDivisionError: division by zero\n帮我修复这个 BUG\n\ndef divide(a, b):\n    return a / b"
回答：<think> 用户提供了报错信息，我应该直接帮他修改</think>
<|EDIT|>
我会使用编辑模式进行处理{"name": "editor", "arguments": {"code": "def divide(a, b):\\n    try:\\n        return a / b\\n    except ZeroDivisionError:\\n        print(\\"Error: Division by zero.\\")\\n        return None"}}

示例2（无报错信息用<|AGENT|>）：
用户："帮我修复这个代码中的 BUG\n\ndef add(a, b):\n    return a - b"
回答：<think> 用户没有直接告诉我 BUG 是什么，所以我需要先调试代码再进行分析，我应该使用代理模式进行尝试</think>
<|AGENT|>
我会使用代理模式进行处理{"name": "python", "arguments": {"code": "def add(a, b):\\n    return a - b"}}
"""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=tools
    )
    
    return text

# 处理所有查询并生成输出
print("=== 开始处理查询 ===")

# 第一步：为所有查询生成prompt
print("正在生成所有查询的prompt...")
text_list = []
for i, query_item in enumerate(queries):
    query = query_item["Query"]
    prompt = generate_prompt(query)
    text_list.append(prompt)

print(f"所有prompt生成完成，共{len(text_list)}个")

# 第二步：批量推理
print("\n开始批量推理...")
start_time = time.time()
outputs = llm.generate(text_list, sampling_params)
end_time = time.time()
inference_time = end_time - start_time
print(f"批量推理完成，耗时: {inference_time:.2f} 秒")

# 第三步：整理结果
print("\n整理结果...")
results = []
for i, (query_item, output) in enumerate(zip(queries, outputs)):
    query = query_item["Query"]
    response = output.outputs[0].text
    
    results.append({
        "Query": query,
        "Output": response
    })
    
# 保存结果到文件
output_file = 'hw3_2.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n=== 处理完成 ===")
print(f"结果已保存到: {output_file}")