#!/usr/bin/env python3
"""
XMLFunctionCallingParser 工具
从 YAML 配置文件加载工具配置并提供解析功能
"""
import os
from pathlib import Path

import yaml
from sweagent.tools.bundle import Bundle

# 导入必要的类和异常
from sweagent.tools.parsing import XMLFunctionCallingParser
from sweagent.tools.tools import ToolConfig


def create_real_tool_config(config_path=""):
    """从 YAML 文件读取配置创建真实的工具配置"""
    """从 YAML 文件读取配置创建真实的工具配置"""
    if not config_path:
        # Path(__file__) 获取当前文件的路径对象。
        # .resolve() 获取绝对路径。
        # .parent 获取父目录。
        # / "tool.yaml" 是拼接路径的简洁写法。
        config_path = Path(__file__).resolve().parent / "tool.yaml"

    with open(config_path, encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)

    swe_agent_root = Path("/mnt/bn/tiktok-mm-5/aiic/users/yiming/SWE/SWE-agent")

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


config = create_real_tool_config()
parser = config.parse_function
commands = config.commands


def xml_parser(model_response):
    """解析模型响应，提取思考、动作和函数调用"""
    if isinstance(model_response, str):
        model_response = {"message": model_response}
    thought, action, function = parser(model_response, commands)
    return thought, action, function


if __name__ == "__main__":
    # 示例用法
    print("加载 XMLFunctionCallingParser...")

    try:
        config_path = Path("./tool.yaml")

        config = create_real_tool_config(config_path)
        parser = config.parse_function
        commands = config.commands
        print(f"✓ 成功加载解析器，共有 {len(commands)} 个可用命令")
        # 显示可用命令
        print("\n📋 可用的命令:")
        for i, cmd in enumerate(commands, 1):
            print(f"{i}. {cmd.name}")

        model_response = {
            "message": "I'll help you implement the necessary changes to fix the issue with the `InheritDocstrings` metaclass not working for properties. Let's follow the steps you outlined.\n\n## Step 1: Find and read code relevant to the PR description\n\nFirst, let's explore the repository structure to locate the `InheritDocstrings` metaclass mentioned in the PR description.\n\n<function=bash>\n<parameter=command>find /testbed -type f -name \"*.py\" | grep -v \"__pycache__\" | sort</parameter>\n</function>## Step 2: Find and read code relevant to the PR description\n\nFirst, let's explore the repository structure to locate the `InheritDocstrings` metaclass mentioned in the PR description.\n\n<function=bash>\n<parameter=command>sort</parameter>\n</function>"
        }

        thought, action, function = parser(model_response, commands)
        print(f"思考: {thought}")
        print(f"动作: {action}")
        print(function)

    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback

        traceback.print_exc()
