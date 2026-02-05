import os

import numpy as np

from .function_parser import xml_parser
from .function_reward import calculate_reward

MAX_SCORE_IN_FUNC_N = 3


def get_weights_gaussian_optimal(arr_len):
    """最佳高斯权重配置"""
    if arr_len <= 1:
        return np.array([1.0] * arr_len)

    positions = np.linspace(0, 1, arr_len)
    peak_position = 0.9
    sigma = 0.05
    return np.exp(-0.5 * ((positions - peak_position) / sigma) ** 2)


def calculate_optimal_weighted_score(avg_p_reward_list):
    """使用最佳配置计算weighted_score"""
    if not avg_p_reward_list or len(avg_p_reward_list) == 0:
        return 0.0

    arr = np.array(avg_p_reward_list)

    # 最佳缩放因子
    clip_lower = 0.05
    clip_upper = 0.95

    # 数据预处理：裁剪和缩放
    arr = np.clip(arr, clip_lower, clip_upper)
    arr = (arr - clip_lower) / (clip_upper - clip_lower)

    # 获取最佳高斯权重
    weights = get_weights_gaussian_optimal(len(arr))
    # 计算加权平均
    weighted_score = np.average(arr, weights=weights)

    return weighted_score


def compute_score_function(solution_str, extra_info):
    try:
        thought, action, parsed_function = xml_parser(solution_str)
    except:
        return {
            "current_score": 0.0,
            "max_score": 0.0,
            "extra_info": {"is_filter": 0, "current_score": 0.0, "max_score": 0.0},
        }

    function = extra_info["function"]
    function_after = extra_info["function_after"]
    current_reward = calculate_reward(parsed_function=parsed_function, target_function=function)
    after_reward = [calculate_reward(parsed_function=parsed_function, target_function=f) for f in function_after]
    all_rewards = [current_reward] + after_reward
    parsed_name = parsed_function.get('name', '')
    parsed_args = parsed_function.get('arguments', {})
    parsed_command = parsed_args.get('command', '')

    if parsed_name == "str_replace_editor" and parsed_command in ["create", "str_replace", "insert", "undo_edit"]:
        max_reward = max(all_rewards[:MAX_SCORE_IN_FUNC_N])
    else:
        max_reward = current_reward
    return {
        "current_score": current_reward,
        "max_score": max_reward,
        "extra_info": {"is_filter": 0, "current_score": current_reward, "max_score": max_reward},
    }
