import re

def format_reward(completions, ground_truth):
    rewards = []
    for comp, gt in zip(completions, ground_truth):
        # 从ground_truth提取答案（匹配末尾####后的数字）
        gt_match = re.search(r'####\s*([\d,]+)\s*$', gt)
        if not gt_match:
            rewards.append(0.0)
            continue
            
        # 清理答案（移除逗号/空格）
        gt_answer = gt_match.group(1).replace(',', '').strip()
        
        # 从completion提取box内容
        comp_match = re.search(r"\\boxed\{(.*?)\}", comp)
        if not comp_match:
            rewards.append(0.0)
            continue
            
        # 清理预测答案
        comp_answer = comp_match.group(1).replace(',', '').strip()
        
        # 判断奖励：格式正确1分，内容正确2分
        if comp_answer == gt_answer:
            rewards.append(2.0)  # 格式+内容正确
        else:
            rewards.append(1.0)  # 仅格式正确
    
    return rewards

def len_reward(completions, ground_truth):
    rewards = []
    for comp, gt in zip(completions, ground_truth):
        len_comp = len(comp)
        len_gt = len(gt)
        lower = 0.8 * len_gt
        upper = 2 * len_gt
        # 长度在合理区间得1分
        rewards.append(1.0 if lower <= len_comp <= upper else 0.0)
    return rewards

def reward_func(completions, ground_truth, **kwargs):
    # 综合格式奖励和长度奖励
    format_scores = format_reward(completions, ground_truth)
    len_scores = len_reward(completions, ground_truth)
    return [f + l for f, l in zip(format_scores, len_scores)]

def compute_metrics(eval_preds):
    generated_answers, references = eval_preds
    # 计算格式和内容完全正确的比例
    format_scores = format_reward(generated_answers, references)
    correct = sum(1 for score in format_scores if score == 2.0)
    total = len(generated_answers) if generated_answers else 1  # 避免除以0
    return {"accuracy": correct / total}