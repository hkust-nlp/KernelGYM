"""
Agent module with base classes and implementations.
"""

from verl_patch.workers.code.agent import BaseAgent
from verl_patch.workers.code.agent.code_agent import CodeAgent
from verl_patch.workers.code.agent.file_search_agent import FileSearchAgent
from verl_patch.workers.code.agent.math_agent import MathAgent
from verl_patch.workers.code.agent.math_neural_interpreter_agent import MathNeuralInterpreterAgent
from verl_patch.workers.code.agent.search_agent import SearchAgent
from .kernel_agent import KernelAgent
# Export all agent classes
__all__ = [
    "BaseAgent",
    "MathAgent",
    "MathNeuralInterpreterAgent",
    "CodeAgent",
    "SearchAgent",
    "FileSearchAgent",
    "create_agent",
    "KernelAgent",
]


def create_agent(agent_type: str, tokenizer) -> BaseAgent:
    """Factory function to create agent instances based on configuration.

    Args:
        agent_type: Type of agent to create (e.g., 'MathAgent', 'CodeAgent')
        tokenizer: Tokenizer instance

    Returns:
        BaseAgent instance

    Raises:
        ValueError: If agent_type is not supported
    """
    agent_map = {
        'MathAgent': MathAgent,
        'MathNeuralInterpreterAgent': MathNeuralInterpreterAgent,
        'CodeAgent': CodeAgent,
        'SearchAgent': SearchAgent,
        'FileSearchAgent': FileSearchAgent,
        'KernelAgent': KernelAgent,
    }

    if agent_type not in agent_map:
        raise ValueError(f"Unsupported agent type: {agent_type}. Supported types: {list(agent_map.keys())}")

    return agent_map[agent_type](tokenizer)
