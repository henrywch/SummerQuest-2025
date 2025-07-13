from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import random

# 使用 ModelScope 加载模型
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 系统提示模板
SYS_PROMPT = """你是一个能进行复杂推理的AI助手，按以下流程处理问题：
1. 在<think>标签内进行多角度深度思考
2. 当需要实时数据/未知知识时调用搜索
3. 搜索调用格式：<|SEARCH|>"""

# 扩展问题池 - 包含需要搜索和不需要搜索的复杂问题
question_pool = [
    # 不需要搜索的问题 (理论/概念性)
    "量子纠缠如何影响量子计算机的错误校正机制？",
    "比较GPT-4和Claude-3在逻辑推理任务中的神经架构差异",
    "如何计算太阳系内小行星轨道共振的稳定性？",
    "解释贝叶斯定理在医疗诊断中的应用及其局限性",
    "深度学习中的注意力机制是如何模拟人类认知过程的？",
    "相对论如何解释GPS系统中的时间膨胀效应？",
    
    # 需要搜索的问题 (实时数据/最新信息)
    "2024年全球芯片短缺对各汽车厂商电动化战略的影响",
    "最新诺贝尔医学奖得主的研究对癌症治疗的潜在影响",
    "当前国际空间站正在进行哪些重要实验？",
    "2024年全球可再生能源投资趋势与主要参与国家",
    "最近三个月主要央行货币政策变化及其对全球经济的影响",
    "当前全球AI芯片市场竞争格局与主要厂商的技术路线",
    
    # 边界情况 (可能需要搜索)
    "巴黎圣母院修复工程的最新进展",
    "量子计算领域最近突破性进展",
    "全球气候变化对主要粮食产区的影响预测"
]

# 多样性参数配置
diversity_params = {
    "temperature": [0.5, 0.7, 0.9],
    "top_p": [0.8, 0.9, 0.95],
    "repetition_penalty": [1.0, 1.1, 1.2],
    "max_new_tokens": [256, 384, 512]
}

# 多阶段生成函数
def generate_data(question):
    # 第一阶段：生成思考内容
    stage1_prompt = (
        f"问题：{question}\n"
        "请进行深入思考，输出格式：<think>你的思考内容</think>\n"
        "思考："
    )
    
    # 随机选择多样性参数
    params = {
        "temperature": random.choice(diversity_params["temperature"]),
        "top_p": random.choice(diversity_params["top_p"]),
        "repetition_penalty": random.choice(diversity_params["repetition_penalty"]),
        "max_new_tokens": random.choice(diversity_params["max_new_tokens"])
    }
    
    # 生成思考内容
    inputs = tokenizer(stage1_prompt, return_tensors="pt").to(model.device)
    stage1_output = model.generate(
        **inputs,
        max_new_tokens=params["max_new_tokens"],
        temperature=params["temperature"],
        top_p=params["top_p"],
        repetition_penalty=params["repetition_penalty"],
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    thoughts = tokenizer.decode(stage1_output[0], skip_special_tokens=True)
    thoughts = thoughts.split("思考：")[-1].strip()
    
    # 确保思考内容在<think>标签中
    if not thoughts.startswith("<think>"):
        thoughts = f"<think>{thoughts}</think>"
    elif not thoughts.endswith("</think>"):
        thoughts = f"{thoughts}</think>"
    
    # 第二阶段：判断是否需要搜索
    stage2_prompt = (
        f"{SYS_PROMPT}\n\n问题：{question}\n"
        f"思考内容：{thoughts}\n"
        "基于以上思考，是否需要搜索最新信息？\n"
        "如果需要搜索，输出：<|SEARCH|>\n"
        "如果不需要搜索，输出：\n"
        "判断："
    )
    
    inputs = tokenizer(stage2_prompt, return_tensors="pt").to(model.device)
    stage2_output = model.generate(
        **inputs,
        max_new_tokens=10,  # 只需短输出判断
        temperature=0.3,    # 低温度确保确定性
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    search_decision = tokenizer.decode(stage2_output[0], skip_special_tokens=True)
    search_decision = search_decision.split("判断：")[-1].strip()
    
    # 处理搜索决策
    search_token = ""
    if "<|SEARCH|>" in search_decision:
        search_token = " <|SEARCH|>"
    else:
        # 验证决策 - 如果思考中提到需要最新数据但仍未添加搜索标记，则强制添加
        if any(keyword in thoughts for keyword in ["最新", "当前", "2024", "实时", "未知"]):
            search_token = " <|SEARCH|>"
    
    # 最终输出组合
    final_output = f"{thoughts}{search_token}"
    
    return {
        "instruction": "处理复杂问题并决定搜索需求",
        "input": question,
        "output": final_output,
        "system": SYS_PROMPT,
        "history": []
    }

# 批量生成数据
dataset = []
for i, question in enumerate(question_pool):
    print(f"生成数据 {i+1}/{len(question_pool)}: {question[:50]}...")
    try:
        data = generate_data(question)
        dataset.append(data)
    except Exception as e:
        print(f"生成问题时出错: {question} - {str(e)}")
    
# 保存为Alpaca格式
with open("complex_qa_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"数据集生成完成，共 {len(dataset)} 条样本")