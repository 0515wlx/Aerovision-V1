"""
VLM (Vision Language Model) 模块
提供GLM-4V API集成功能，包括：
- GLM客户端 (glm_client)
- Prompt工程 (prompts)
- 推理接口 (inference)
"""

from .glm_client import GLMClient, GLMResponse
from .prompts import PromptEngine, PromptResult, get_prompt, list_tasks
from .inference import VLMInference, InferenceResult, create_inference, infer

__all__ = [
    # GLM客户端
    "GLMClient",
    "GLMResponse",
    # Prompt引擎
    "PromptEngine",
    "PromptResult",
    "get_prompt",
    "list_tasks",
    # 推理引擎
    "VLMInference",
    "InferenceResult",
    "create_inference",
    "infer",
]
