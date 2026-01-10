"""
GLM-4V API客户端
提供与智谱AI GLM-4V模型的交互接口
"""

import os
import base64
import json
import time
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path

import requests
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


@dataclass
class GLMResponse:
    """GLM API响应结果"""
    success: bool
    content: Optional[str] = None
    parsed_data: Optional[Dict[str, Any]] = None
    raw_response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class GLMClient:
    """GLM-4V API客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model: str = "glm-4v",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        top_p: float = 0.7,
        timeout_connect: int = 30,
        timeout_read: int = 120,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
    ):
        """
        初始化GLM客户端

        Args:
            api_key: API密钥，如果为None则从环境变量GLM_API_KEY读取
            api_url: API端点URL，如果为None则从环境变量GLM_API_URL读取
            model: 模型名称，默认为glm-4v
            temperature: 温度参数
            max_tokens: 最大生成token数
            top_p: Top-p采样参数
            timeout_connect: 连接超时时间（秒）
            timeout_read: 读取超时时间（秒）
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            backoff_factor: 指数退避因子
        """
        self.api_key = api_key or os.getenv("GLM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API密钥未提供。请设置GLM_API_KEY环境变量或传入api_key参数"
            )

        self.api_url = api_url or os.getenv(
            "GLM_API_URL",
            "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout_connect = timeout_connect
        self.timeout_read = timeout_read
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor

        # 设置日志
        self.logger = logging.getLogger(__name__)

    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        将图片文件编码为base64字符串

        Args:
            image_path: 图片文件路径

        Returns:
            str: base64编码的图片字符串
        """
        with open(image_path, "rb") as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode("utf-8")

    def _encode_image_bytes_to_base64(self, image_bytes: bytes) -> str:
        """
        将图片字节数据编码为base64字符串

        Args:
            image_bytes: 图片字节数据

        Returns:
            str: base64编码的图片字符串
        """
        return base64.b64encode(image_bytes).decode("utf-8")

    def _prepare_messages(
        self,
        images: Union[str, List[str], bytes, List[bytes]],
        system_prompt: str,
        user_prompt: str
    ) -> List[Dict[str, Any]]:
        """
        准备API请求的消息格式

        Args:
            images: 图片路径或字节数据，支持单个或多个
            system_prompt: 系统提示词
            user_prompt: 用户提示词

        Returns:
            List[Dict[str, Any]]: 格式化的消息列表
        """
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

        # 处理图片输入
        if isinstance(images, (str, bytes)):
            images = [images]

        image_contents = []
        for img in images:
            if isinstance(img, str):
                # 文件路径
                base64_image = self._encode_image_to_base64(img)
                image_contents.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
            elif isinstance(img, bytes):
                # 字节数据
                base64_image = self._encode_image_bytes_to_base64(img)
                image_contents.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })

        # 添加用户消息（包含图片和文本）
        user_content = [
            {"type": "text", "text": user_prompt}
        ]
        user_content.extend(image_contents)

        messages.append({
            "role": "user",
            "content": user_content
        })

        return messages

    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """
        尝试解析响应中的JSON内容

        Args:
            content: API返回的文本内容

        Returns:
            Optional[Dict[str, Any]]: 解析后的JSON数据，失败返回None
        """
        # 尝试直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 尝试提取JSON代码块
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试提取花括号内的内容
        brace_match = re.search(r'\{.*\}', content, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _make_request(
        self,
        messages: List[Dict[str, Any]],
        attempt: int = 1
    ) -> Dict[str, Any]:
        """
        发起API请求（带重试机制）

        Args:
            messages: 消息列表
            attempt: 当前尝试次数

        Returns:
            Dict[str, Any]: API响应数据

        Raises:
            requests.RequestException: 请求失败
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p
        }

        self.logger.info(f"发送GLM API请求（尝试 {attempt}/{self.max_retries}）")

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=(self.timeout_connect, self.timeout_read)
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout as e:
            self.logger.warning(f"请求超时: {e}")
            raise
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP错误: {e}")
            raise
        except requests.exceptions.RequestException as e:
            self.logger.error(f"请求异常: {e}")
            raise

    def chat(
        self,
        images: Union[str, List[str], bytes, List[bytes]],
        system_prompt: str,
        user_prompt: str,
        parse_json: bool = True
    ) -> GLMResponse:
        """
        与GLM-4V模型进行对话

        Args:
            images: 图片路径或字节数据，支持单个或多个
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            parse_json: 是否尝试解析响应中的JSON

        Returns:
            GLMResponse: 响应结果
        """
        messages = self._prepare_messages(images, system_prompt, user_prompt)

        last_error = None
        raw_response = None

        for attempt in range(1, self.max_retries + 1):
            try:
                raw_response = self._make_request(messages, attempt)
                break
            except requests.RequestException as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = self.retry_delay * (self.backoff_factor ** (attempt - 1))
                    self.logger.info(f"等待 {delay:.1f}秒后重试...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"达到最大重试次数 {self.max_retries}，放弃请求")

        if raw_response is None:
            return GLMResponse(
                success=False,
                error=f"API请求失败: {str(last_error)}",
                metadata={"attempts": self.max_retries}
            )

        # 解析响应
        try:
            content = raw_response["choices"][0]["message"]["content"]
            parsed_data = None

            if parse_json:
                parsed_data = self._parse_json_response(content)

            return GLMResponse(
                success=True,
                content=content,
                parsed_data=parsed_data,
                raw_response=raw_response,
                metadata={
                    "model": raw_response.get("model"),
                    "usage": raw_response.get("usage"),
                    "attempts": attempt
                }
            )

        except (KeyError, IndexError) as e:
            return GLMResponse(
                success=False,
                error=f"响应解析失败: {str(e)}",
                raw_response=raw_response
            )

    def batch_chat(
        self,
        image_list: List[Union[str, bytes]],
        system_prompt: str,
        user_prompt: str,
        parse_json: bool = True,
        concurrent: int = 3,
        delay: float = 0.1
    ) -> List[GLMResponse]:
        """
        批量对话处理

        Args:
            image_list: 图片路径或字节数据列表
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            parse_json: 是否尝试解析响应中的JSON
            concurrent: 并发请求数
            delay: 请求间隔（秒）

        Returns:
            List[GLMResponse]: 响应结果列表
        """
        import concurrent.futures

        results = []
        total = len(image_list)

        self.logger.info(f"开始批量处理 {total} 张图片，并发数: {concurrent}")

        def process_image(idx: int, image: Union[str, bytes]) -> tuple:
            time.sleep(delay * idx)  # 错开请求时间
            result = self.chat(image, system_prompt, user_prompt, parse_json)
            return idx, result

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent) as executor:
            futures = [
                executor.submit(process_image, idx, img)
                for idx, img in enumerate(image_list)
            ]

            for future in concurrent.futures.as_completed(futures):
                idx, result = future.result()
                results.append((idx, result))
                self.logger.info(f"完成 {idx + 1}/{total}")

        # 按原始顺序排序
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GLMClient":
        """
        从配置字典创建客户端实例

        Args:
            config: 配置字典

        Returns:
            GLMClient: 客户端实例
        """
        glm_config = config.get("glm", {})
        generation = glm_config.get("generation", {})
        timeout = glm_config.get("timeout", {})
        retry = glm_config.get("retry", {})

        return cls(
            api_url=glm_config.get("api_url"),
            model=glm_config.get("model", "glm-4v"),
            temperature=generation.get("temperature", 0.3),
            max_tokens=generation.get("max_tokens", 1024),
            top_p=generation.get("top_p", 0.7),
            timeout_connect=timeout.get("connect", 30),
            timeout_read=timeout.get("read", 120),
            max_retries=retry.get("max_attempts", 3),
            retry_delay=retry.get("delay", 1.0),
            backoff_factor=retry.get("backoff_factor", 2.0),
        )
