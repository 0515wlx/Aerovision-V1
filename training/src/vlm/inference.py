"""
VLM推理模块
提供统一的推理接口，支持批量处理和标准化输出
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from .glm_client import GLMClient, GLMResponse
from .prompts import PromptEngine, PromptResult, get_prompt


@dataclass
class InferenceResult:
    """推理结果"""
    task_type: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    content: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class VLMInference:
    """VLM推理引擎"""

    def __init__(
        self,
        client: Optional[GLMClient] = None,
        prompt_engine: Optional[PromptEngine] = None
    ):
        """
        初始化VLM推理引擎

        Args:
            client: GLM客户端实例，如果为None则创建默认客户端
            prompt_engine: Prompt引擎实例，如果为None则创建默认引擎
        """
        self.client = client or GLMClient()
        self.prompt_engine = prompt_engine or PromptEngine()
        self.logger = logging.getLogger(__name__)

    def infer(
        self,
        images: Union[str, List[str], bytes, List[bytes]],
        task_type: str,
        **prompt_kwargs
    ) -> InferenceResult:
        """
        执行单次推理

        Args:
            images: 图片路径或字节数据，支持单个或多个
            task_type: 任务类型 (aircraft_type, airline, registration, quality)
            **prompt_kwargs: prompt特定参数

        Returns:
            InferenceResult: 推理结果
        """
        try:
            # 获取prompt
            prompt_result = self.prompt_engine.get_prompt(task_type, **prompt_kwargs)

            self.logger.info(f"执行 {task_type} 推理任务")

            # 调用GLM API
            glm_response = self.client.chat(
                images=images,
                system_prompt=prompt_result.system_prompt,
                user_prompt=prompt_result.user_prompt,
                parse_json=True
            )

            # 构建推理结果
            metadata = {
                "task_type": task_type,
                "prompt_metadata": prompt_result.metadata,
            }

            if glm_response.success:
                metadata.update(glm_response.metadata or {})

                return InferenceResult(
                    task_type=task_type,
                    success=True,
                    data=glm_response.parsed_data,
                    content=glm_response.content,
                    metadata=metadata
                )
            else:
                return InferenceResult(
                    task_type=task_type,
                    success=False,
                    error=glm_response.error,
                    metadata=metadata
                )

        except Exception as e:
            self.logger.error(f"推理失败: {e}", exc_info=True)
            return InferenceResult(
                task_type=task_type,
                success=False,
                error=str(e),
                metadata={"task_type": task_type}
            )

    def batch_infer(
        self,
        image_list: List[Union[str, bytes]],
        task_type: str,
        **prompt_kwargs
    ) -> List[InferenceResult]:
        """
        批量推理

        Args:
            image_list: 图片路径或字节数据列表
            task_type: 任务类型
            **prompt_kwargs: prompt特定参数

        Returns:
            List[InferenceResult]: 推理结果列表
        """
        try:
            # 获取prompt
            prompt_result = self.prompt_engine.get_prompt(task_type, **prompt_kwargs)

            self.logger.info(f"执行 {task_type} 批量推理，共 {len(image_list)} 张图片")

            # 批量调用GLM API
            glm_responses = self.client.batch_chat(
                image_list=image_list,
                system_prompt=prompt_result.system_prompt,
                user_prompt=prompt_result.user_prompt,
                parse_json=True
            )

            # 转换为推理结果
            results = []
            for glm_response in glm_responses:
                metadata = {
                    "task_type": task_type,
                    "prompt_metadata": prompt_result.metadata,
                }
                if glm_response.success:
                    metadata.update(glm_response.metadata or {})
                    results.append(InferenceResult(
                        task_type=task_type,
                        success=True,
                        data=glm_response.parsed_data,
                        content=glm_response.content,
                        metadata=metadata
                    ))
                else:
                    results.append(InferenceResult(
                        task_type=task_type,
                        success=False,
                        error=glm_response.error,
                        metadata=metadata
                    ))

            return results

        except Exception as e:
            self.logger.error(f"批量推理失败: {e}", exc_info=True)
            # 返回失败结果列表
            return [
                InferenceResult(
                    task_type=task_type,
                    success=False,
                    error=str(e),
                    metadata={"task_type": task_type}
                )
                for _ in image_list
            ]

    def infer_aircraft_type(
        self,
        images: Union[str, List[str], bytes, List[bytes]],
        aircraft_types: Optional[List[str]] = None
    ) -> InferenceResult:
        """
        机型识别推理

        Args:
            images: 图片路径或字节数据
            aircraft_types: 可选的机型类别列表

        Returns:
            InferenceResult: 推理结果
        """
        return self.infer(
            images=images,
            task_type="aircraft_type",
            aircraft_types=aircraft_types
        )

    def infer_airline(
        self,
        images: Union[str, List[str], bytes, List[bytes]],
        airlines: Optional[List[str]] = None
    ) -> InferenceResult:
        """
        航司识别推理

        Args:
            images: 图片路径或字节数据
            airlines: 可选的航司类别列表

        Returns:
            InferenceResult: 推理结果
        """
        return self.infer(
            images=images,
            task_type="airline",
            airlines=airlines
        )

    def infer_registration(
        self,
        images: Union[str, List[str], bytes, List[bytes]]
    ) -> InferenceResult:
        """
        注册号OCR推理

        Args:
            images: 图片路径或字节数据

        Returns:
            InferenceResult: 推理结果
        """
        return self.infer(
            images=images,
            task_type="registration"
        )

    def infer_quality(
        self,
        images: Union[str, List[str], bytes, List[bytes]]
    ) -> InferenceResult:
        """
        质量评估推理

        Args:
            images: 图片路径或字节数据

        Returns:
            InferenceResult: 推理结果
        """
        return self.infer(
            images=images,
            task_type="quality"
        )

    def save_results(
        self,
        results: List[InferenceResult],
        output_path: str,
        format: str = "json"
    ) -> None:
        """
        保存推理结果

        Args:
            results: 推理结果列表
            output_path: 输出路径
            format: 输出格式 (json)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            data = [asdict(r) for r in results]
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"不支持的输出格式: {format}")

        self.logger.info(f"结果已保存到 {output_path}")

    def load_results(
        self,
        input_path: str
    ) -> List[InferenceResult]:
        """
        加载推理结果

        Args:
            input_path: 输入路径

        Returns:
            List[InferenceResult]: 推理结果列表
        """
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [InferenceResult(**d) for d in data]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VLMInference":
        """
        从配置字典创建推理引擎实例

        Args:
            config: 配置字典

        Returns:
            VLMInference: 推理引擎实例
        """
        client = GLMClient.from_config(config)
        return cls(client=client)


# 便捷函数
def create_inference(
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    **kwargs
) -> VLMInference:
    """
    创建VLM推理引擎（便捷函数）

    Args:
        api_key: API密钥
        api_url: API端点URL
        **kwargs: 其他GLM客户端参数

    Returns:
        VLMInference: 推理引擎实例
    """
    client = GLMClient(
        api_key=api_key,
        api_url=api_url,
        **kwargs
    )
    return VLMInference(client=client)


def infer(
    images: Union[str, List[str], bytes, List[bytes]],
    task_type: str,
    **kwargs
) -> InferenceResult:
    """
    执行单次推理（便捷函数）

    Args:
        images: 图片路径或字节数据
        task_type: 任务类型
        **kwargs: prompt特定参数

    Returns:
        InferenceResult: 推理结果
    """
    inference = create_inference()
    return inference.infer(images, task_type, **kwargs)
