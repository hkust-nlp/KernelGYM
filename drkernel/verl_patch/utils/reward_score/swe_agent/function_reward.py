#!/usr/bin/env python3
"""
函数奖励测试工具
调用 xml_function_calling_parser.py 解析 responses.json 中的每一条响应
"""

import difflib
import json
import os
import sys
from pathlib import Path

import yaml
from sweagent.tools.bundle import Bundle

# 导入必要的类和异常
from sweagent.tools.parsing import XMLFunctionCallingParser
from sweagent.tools.tools import ToolConfig

from verl_patch.utils.reward_score.swe.patch_similarity import (
    compute_change_similarities,
    generate_unified_diff,
)


def create_real_tool_config(config_path, swe_agent_root):
    """从 YAML 文件读取配置创建真实的工具配置"""
    # 读取 YAML 配置文件
    with open(config_path, encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)

    # 从 YAML 配置中读取 bundles
    bundles = []
    for bundle_config in yaml_config['tools']['bundles']:
        bundle_path = swe_agent_root / bundle_config['path']
        bundles.append(Bundle(path=bundle_path))

    # 创建工具配置
    config = ToolConfig(
        bundles=bundles,
        env_variables=yaml_config['tools']['env_variables'],
        enable_bash_tool=yaml_config['tools']['enable_bash_tool'],
        parse_function=XMLFunctionCallingParser(),
        execution_timeout=yaml_config['tools']['execution_timeout'],
    )

    return config


def calculate_range_similarity(target_range, pred_range):
    """
    计算view_range相似度，考虑空range代表全文件

    Args:
        target_range: 目标范围，如果为空表示没有限制
        pred_range: 预测范围

    Returns:
        float: 相似度分数 (0.0-1.0)
    """

    # 如果target_range没有指定（为空），pred_range是什么都可以
    if not target_range:
        return 1.0

    # 如果target_range指定了，但pred_range为空，需要判断target_range是否表示全文件
    if not pred_range:
        # target_range为空时已经在上面处理了，这里target_range不为空
        # pred_range为空表示预测的是全文件
        if len(target_range) != 2:
            return 0.0

        start, end = target_range[0], target_range[1]

        # 如果target是 [1, -1] 表示从头到尾，与全文件完全相同
        if start == 1 and end == -1:
            return 1.0
        # 如果target是从第1行开始的大范围，给较高相似度
        if start == 1:
            return 0.7
        # 如果target包含 -1（到文件末尾），给中等相似度
        if end == -1:
            return 0.5

        # 其他情况根据target范围大小给相似度
        range_size = end - start + 1
        if range_size >= 50:
            return 0.4
        elif range_size >= 20:
            return 0.3
        else:
            return 0.2

    # 两个都不为空 - 计算Jaccard相似度
    if len(target_range) != 2 or len(pred_range) != 2:
        return 0.0

    target_start, target_end = target_range[0], target_range[1]
    pred_start, pred_end = pred_range[0], pred_range[1]

    ASSUMED_FILE_LENGTH = max(target_end, pred_end, target_start, pred_start, 100) * 2
    if target_end == -1:
        target_end = ASSUMED_FILE_LENGTH
    if pred_end == -1:
        pred_end = ASSUMED_FILE_LENGTH

    # 计算Jaccard相似度
    intersection_start = max(target_start, pred_start)
    intersection_end = min(target_end, pred_end)
    intersection_size = max(0, intersection_end - intersection_start + 1)

    union_start = min(target_start, pred_start)
    union_end = max(target_end, pred_end)
    union_size = union_end - union_start + 1

    return intersection_size / union_size if union_size > 0 else 0.0


def calculate_reward(parsed_function, target_function):
    """计算函数奖励
    Args:
        parsed_function: 解析得到的函数调用
        target_function: 目标函数（来自functions.json）
    Returns:
        如果函数名不同返回0；如果参数键不同返回0；如果参数键相同则根据值的相似度计算平均分数
    """
    if not parsed_function or not target_function:
        return 0

    # 检查函数名是否相等
    parsed_name = parsed_function.get('name', '')
    target_name = target_function.get('name', '')

    parsed_args = parsed_function.get('arguments', {})
    target_args = target_function.get('arguments', {})

    if isinstance(parsed_args, str):
        parsed_args = json.loads(parsed_args)
    if isinstance(target_args, str):
        target_args = json.loads(target_args)

    if parsed_name != target_name:
        return 0
    if parsed_name == 'bash':
        parsed_command = parsed_args.get('command', '')
        target_command = target_args.get('command', '')
        # 按空格分割为tokens，这样能更好地处理命令结构
        parsed_tokens = parsed_command.split()
        target_tokens = target_command.split()

        # 比较token序列，这样参数顺序变化影响较小
        matcher = difflib.SequenceMatcher(None, parsed_tokens, target_tokens)
        similarity = matcher.ratio()
        return similarity
    elif parsed_name == 'submit':
        return 1.0  # submit函数不需要参数比较，直接返回1.0
    elif parsed_name == 'str_replace_editor':
        parsed_command = parsed_args.get('command', '')
        target_command = target_args.get('command', '')
        if parsed_command != target_command:
            return 0.0
        # ['view', 'create', 'str_replace', 'insert', 'undo_edit']
        if parsed_command == 'view':
            if parsed_args.get('path') != target_args.get('path'):
                return 0.0
            else:
                target_view_range = target_args.get('view_range', [])
                parsed_view_range = parsed_args.get('view_range', [])
                # 计算两个range Jaccard 相似度, 暂不用
                similarity = calculate_range_similarity(target_view_range, parsed_view_range)
                return similarity
        elif parsed_command == 'create':
            target_file_text = target_args.get('file_text', '')
            parsed_file_text = parsed_args.get('file_text', '')
            similarity = difflib.SequenceMatcher(None, parsed_file_text, target_file_text, autojunk=False).ratio()
            return similarity
        elif parsed_command == 'str_replace':
            if parsed_args.get('path') != target_args.get('path'):
                return 0.0
            path = parsed_args.get('path', '')
            parsed_old_str = parsed_args.get('old_str', '')
            target_old_str = target_args.get('old_str', '')
            parsed_new_str = parsed_args.get('new_str', '')
            target_new_str = target_args.get('new_str', '')

            parsed_patch = generate_unified_diff(parsed_old_str, parsed_new_str, n_context=1)
            target_patch = generate_unified_diff(target_old_str, target_new_str, n_context=1)
            # 计算补丁的相似度
            similarity = compute_change_similarities({path: parsed_patch}, {path: target_patch})[0]["similarity"]

            return similarity
        elif parsed_command == 'insert':
            if parsed_args.get('path') != target_args.get('path'):
                return 0.0
            if parsed_args.get('insert_line') != target_args.get('insert_line'):
                return 0.0
            target_file_text = target_args.get('file_text', '')
            parsed_file_text = parsed_args.get('file_text', '')
            similarity = difflib.SequenceMatcher(None, parsed_file_text, target_file_text, autojunk=False).ratio()
            return similarity
        elif parsed_command == 'undo_edit':
            return parsed_args.get('path') == target_args.get('path')
        else:
            raise ValueError(f"Unsupported command: {parsed_command}")
    else:
        raise ValueError(f"Unsupported function: {parsed_name}")
