"""
Agent module with base classes and implementations.
"""

from .base_agent import BaseAgent
from .code_agent import CodeAgent
from .file_search_agent import FileSearchAgent
from .math_agent import MathAgent
from .math_neural_interpreter_agent import MathNeuralInterpreterAgent
from .search_agent import SearchAgent

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
