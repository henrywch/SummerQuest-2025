import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

print("-" * 50)
print("""评分标准：
    满分 100 分
    1. 文件交全 10 分
    2. 自我介绍 10 分
    3. hw1 能体现搜查关键字 30 分
    4. hw2_1 共 10 分
    5. hw2_2 共 20 分，每个环境输出占一定分数
    6. hw2_3 共 20 分, 每生成一个回复 10 分 """   
)
print('-' * 50)

def file_overall_check():
    '''
    检查所有文件是否齐全
    '''
    score = 10
    
    file_names = ["README.md", "hw1.log", "hw2_1.log", "hw2_2.log", "hw2_3.log"]

    self_intro_path = os.path.join(parent_dir, "README.md")
    if not os.path.isfile(self_intro_path):
        print("未找到自我介绍!")
        score = 0
        
    for file_name in file_names:
        file_path = os.path.join(current_dir, file_name)

        if not os.path.isfile(file_path):
            print(f"未找到{file_name}！")
            score = 0
        
    return score
    
def check_self_intro():
    '''
    检查自我介绍
    要求: 有一个标题
    '''
    
    self_intro_path = os.path.join(parent_dir, "README.md")
    
    if not os.path.isfile(self_intro_path):
        return 0
    
    with open(self_intro_path, 'r', encoding='utf-8') as f:
        content = f.read()
         
    # print("FOR DEBUG")   
    # print(content)
    
    content = content.strip()
    if content[0] != "#":
        return 0 
    else:
        return 10
    
def check_hw1():
    '''
    检查 hw1.log
    要求: 体现了对 \"刘智耿\" 的搜索
    '''
        
    hw1_path = os.path.join(current_dir, "hw1.log")
    
    if not os.path.isfile(hw1_path):
        return 0
    
    with open(hw1_path, 'r', encoding='utf-8') as f:
        content = f.read()    
        
    if "刘智耿" in content:
        return 30
    else:
        return 0
    
def check_hw2_1():
    '''
    检查 hw2_1.log
    要求: 要求出现 NVIDIA-SMI 的表格, 并且有一张 GeForce 显卡
    '''
    hw2_1_path = os.path.join(current_dir, "hw2_1.log")
    
    if not os.path.isfile(hw2_1_path):
        return 0
    
    with open(hw2_1_path, 'r', encoding='utf-8') as f:
        content = f.read()    
        
    if "NVIDIA-SMI" in content and "GeForce" in content:
        return 10
    else:
        return 0
    
def check_hw2_2():
    '''
    检查 hw2_2.log
    要求: 正确地显示了 Pytorch, Cuda, Transformers 的版本, 生成了两个 AI 回复
    '''
    
    hw2_2_path = os.path.join(current_dir, "hw2_2.log")
    
    if not os.path.isfile(hw2_2_path):
        return 0
    
    score = 0
    
    with open(hw2_2_path, 'r', encoding='utf-8') as f:
        content = f.read()  
        
    if "PyTorch 版本" in content:
        score += 4
        
    if "CUDA 版本" in content:
        score += 4
    
    if "Transformers 版本" in content:
        score += 4
        
    if "AI回复（普通模式）:" in content:
        score += 4
        
    if "AI回复（思维链模式）:" in content:
        score += 4
        
    return score

def check_hw2_3():
    '''
    检查 hw2_3.log
    要求: 生成了两个 AI 回复
    '''
    
    hw2_3_path = os.path.join(current_dir, "hw2_3.log")
    
    if not os.path.isfile(hw2_3_path):
        return 0
    
    score = 0
    
    with open(hw2_3_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    if "AI回复（普通模式 - vLLM）:" in content:
        score += 10
        
    if "AI回复（思维链模式 - vLLM）:"  in content:
        score += 10
        
    return score
    
    
if __name__ == '__main__':
    print("检查作业中...")
    score = 0
    
    temp = file_overall_check()
    print(f"文件完整性得分: {temp}/10")
    score += temp
    
    temp = check_self_intro()
    print(f"自我介绍得分: {temp}/10")
    score += temp
    
    temp = check_hw1()
    print(f"飞书检索得分为： {temp}/30")
    score += temp
    
    temp = check_hw2_1()
    print(f"hw2_1 得分为: {temp}/10")
    score += temp
    
    temp = check_hw2_2()
    print(f"hw2_2 得分为: {temp}/20")
    score += temp
    
    temp = check_hw2_3()
    print(f"hw2_3 得分为: {temp}/20")
    score += temp
    
    print("检查完毕！")
    
    print('-' * 50)
    print(f"你的总分为: {score}/100")
    
    if score == 100:
        print("你得了 M↓V↑P↑ !")
    print('-' * 50)
    
