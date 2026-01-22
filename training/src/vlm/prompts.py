"""
VLM Prompt工程模块
为GLM-4V API设计各种识别任务的prompt模板
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PromptResult:
    """Prompt处理结果"""
    system_prompt: str
    user_prompt: str
    task_type: str
    metadata: Dict[str, Any]


class PromptEngine:
    """Prompt工程引擎"""

    def __init__(self):
        self._prompts = {
            "aircraft_type": self._get_aircraft_type_prompt,
            "airline": self._get_airline_prompt,
            "registration": self._get_registration_prompt,
            "quality": self._get_quality_prompt,
        }

    def get_prompt(
        self,
        task_type: str,
        **kwargs
    ) -> PromptResult:
        """
        获取指定任务的prompt

        Args:
            task_type: 任务类型 (aircraft_type, airline, registration, quality)
            **kwargs: 任务特定参数

        Returns:
            PromptResult: 包含system_prompt和user_prompt的结果对象
        """
        if task_type not in self._prompts:
            raise ValueError(f"未知任务类型: {task_type}. 可用类型: {list(self._prompts.keys())}")

        prompt_func = self._prompts[task_type]
        return prompt_func(**kwargs)

    def _get_aircraft_type_prompt(
        self,
        aircraft_types: Optional[List[str]] = None,
        **kwargs
    ) -> PromptResult:
        """
        获取机型识别prompt

        Args:
            aircraft_types: 可选的机型类别列表
            **kwargs: 其他参数

        Returns:
            PromptResult: 机型识别prompt
        """
        system_prompt = """你是一个专业的航空器识别专家。你的任务是从提供的飞机图片中准确识别飞机的机型。

识别要求：
1. 仔细观察飞机的整体外形、机翼形状、发动机位置、尾翼结构等特征
2. 根据提供的机型列表进行识别，选择最匹配的机型
3. 输出机型名称时，请使用英文简称（如 A320, B738, C919 等）
4. 给出识别结果的置信度（0-1之间的数值，1表示完全确定），置信度必须客观准确，确保与实际识别准确率相匹配
5. 请注意：当前模型在该任务上的准确率约为53%，请客观评估你的识别结果，不要过度自信
6. 输出的机型必须是data/aircraft_types.json文件中定义的英文简称，如果识别的机型不在该文件中，请输出"unknown"

输出格式要求：
请严格按照以下JSON格式输出，不要包含任何其他内容：
{
    "aircraft_type": "A320",
    "confidence": 0.6,
    "reasoning": "识别理由简述"
}

示例：
- 对于清晰的空客A320图片：{"aircraft_type": "A320", "confidence": 0.85, "reasoning": "机翼形状和发动机位置符合A320特征"}
- 对于模糊的波音737图片：{"aircraft_type": "B738", "confidence": 0.4, "reasoning": "图片模糊，但发动机位置符合737系列特征"}
- 对于未在列表中的机型：{"aircraft_type": "unknown", "confidence": 0.3, "reasoning": "该机型不在预定义列表中"}"""

        if aircraft_types:
            types_str = "、".join(aircraft_types)
            user_prompt = f"""请识别这张图片中的飞机机型。

可选机型列表：{types_str}

请根据飞机的外观特征，从上述列表中选择最匹配的机型（使用英文简称），并给出客观的置信度和识别理由。注意：输出的机型必须是列表中的英文简称，如果识别的机型不在列表中，请输出"unknown"。"""
        else:
            user_prompt = """请识别这张图片中的飞机机型。

请根据飞机的外观特征识别机型（使用英文简称），并给出客观的置信度和识别理由。注意：输出的机型必须是data/aircraft_types.json文件中定义的英文简称，如果识别的机型不在该文件中，请输出"unknown"。"""

        return PromptResult(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            task_type="aircraft_type",
            metadata={"aircraft_types": aircraft_types}
        )

    def _get_airline_prompt(
        self,
        airlines: Optional[List[str]] = None,
        **kwargs
    ) -> PromptResult:
        """
        获取航司识别prompt

        Args:
            airlines: 可选的航司类别列表
            **kwargs: 其他参数

        Returns:
            PromptResult: 航司识别prompt
        """
        system_prompt = """你是一个专业的航空公司识别专家。你的任务是从提供的飞机图片中准确识别飞机所属的航空公司。

识别要求：
1. 仔细观察飞机的涂装、logo、机身文字、尾翼图案等特征
2. 根据提供的航司列表进行识别，选择最匹配的航司
3. 输出航空公司名称时，请使用英文简称（航司代码，如 CEA, CCA, CSN 等）
4. 给出识别结果的置信度（0-1之间的数值，1表示完全确定），置信度必须客观准确，确保与实际识别准确率相匹配
5. 请注意：当前模型在该任务上的准确率约为53%，请客观评估你的识别结果，不要过度自信
6. 输出的航司必须是data/airlines.json文件中定义的英文简称，如果识别的航司不在该文件中，请输出"unknown"

输出格式要求：
请严格按照以下JSON格式输出，不要包含任何其他内容：
{
    "airline": "CEA",
    "confidence": 0.6,
    "reasoning": "识别理由简述"
}

示例：
- 对于清晰的中国东方航空涂装：{"airline": "CEA", "confidence": 0.8, "reasoning": "红蓝涂装和燕子logo符合中国东方航空特征"}
- 对于模糊的国航涂装：{"airline": "CCA", "confidence": 0.45, "reasoning": "图片模糊，但凤凰logo隐约可见，可能为国航"}
- 对于未在列表中的航司：{"airline": "unknown", "confidence": 0.3, "reasoning": "该航司不在预定义列表中"}"""

        if airlines:
            airlines_str = "、".join(airlines)
            user_prompt = f"""请识别这张图片中的飞机所属的航空公司。

可选航司列表：{airlines_str}

请根据飞机的涂装和logo特征，从上述列表中选择最匹配的航司（使用英文简称），并给出客观的置信度和识别理由。注意：输出的航司必须是列表中的英文简称，如果识别的航司不在列表中，请输出"unknown"。"""
        else:
            user_prompt = """请识别这张图片中的飞机所属的航空公司。

请根据飞机的涂装和logo特征识别航司（使用英文简称），并给出客观的置信度和识别理由。注意：输出的航司必须是data/airlines.json文件中定义的英文简称，如果识别的航司不在该文件中，请输出"unknown"。"""

        return PromptResult(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            task_type="airline",
            metadata={"airlines": airlines}
        )

    def _get_registration_prompt(
        self,
        **kwargs
    ) -> PromptResult:
        """
        获取OCR注册号识别prompt

        Args:
            **kwargs: 其他参数

        Returns:
            PromptResult: 注册号识别prompt
        """
        system_prompt = """你是一个专业的航空器注册号OCR识别专家。你的任务是从提供的飞机注册号区域图片中准确识别注册号文字。

识别要求：
1. 仔细观察图片中的字母和数字，注意区分相似的字符（如O和0、I和1等）
2. 注册号格式通常为：B-XXXX（中国）、N-XXXX（美国）、G-XXXX（英国）等
3. 给出识别结果的置信度（0-1之间的数值，1表示完全确定），置信度必须客观准确，确保与实际识别准确率相匹配
4. 如果图片模糊或无法识别，请给出最可能的识别结果并标注低置信度

输出格式要求：
请严格按照以下JSON格式输出，不要包含任何其他内容：
{
    "registration": "B-1234",
    "confidence": 0.6,
    "reasoning": "识别理由简述"
}

示例：
- 对于清晰的注册号图片：{"registration": "B-1234", "confidence": 0.9, "reasoning": "字符清晰可辨"}
- 对于模糊的注册号图片：{"registration": "B-1X34", "confidence": 0.5, "reasoning": "第三位字符模糊，可能是2或X"}"""

        user_prompt = """请识别这张图片中的飞机注册号。

请仔细观察图片中的字母和数字，准确识别注册号，并给出客观的置信度和识别理由。注意区分相似的字符。"""

        return PromptResult(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            task_type="registration",
            metadata={}
        )

    def _get_quality_prompt(
        self,
        **kwargs
    ) -> PromptResult:
        """
        获取质量评估prompt

        Args:
            **kwargs: 其他参数

        Returns:
            PromptResult: 质量评估prompt
        """
        system_prompt = """你是一个专业的图像质量评估专家。你的任务是对提供的飞机图片进行质量评估。

评估要求：
1. 清晰度评分（clarity）：评估图片的整体清晰度，范围0.0-1.0
2. 遮挡程度评分（block）：评估飞机被遮挡的程度，范围0.0-1.0
3. 给出评估结果的置信度（0-1之间的数值，1表示完全确定），置信度必须客观准确，确保与实际评估准确率相匹配

清晰度评分标准：
- 0.9-1.0：非常清晰（细节锐利，可看清小字）
- 0.7-0.9：清晰（整体清晰，细节略有模糊）
- 0.5-0.7：一般（能辨认机型，但不够锐利）
- 0.3-0.5：模糊（勉强能辨认）
- 0.0-0.3：非常模糊（几乎无法辨认）

遮挡程度评分标准：
- 0.0：无遮挡（飞机完全可见）
- 0.1-0.3：轻微遮挡（一小部分被遮挡）
- 0.3-0.5：部分遮挡（约1/3被遮挡）
- 0.5-0.7：明显遮挡（约一半被遮挡）
- 0.7-1.0：严重遮挡（大部分被遮挡）

输出格式要求：
请严格按照以下JSON格式输出，不要包含任何其他内容：
{
    "clarity": 0.85,
    "block": 0.17,
    "confidence": 0.66,
    "reasoning": "评估理由简述"
}

示例：
- 对于清晰无遮挡的图片：{"clarity": 0.9, "block": 0.0, "confidence": 0.95, "reasoning": "图片清晰锐利，飞机完全可见"}
- 对于一般清晰度、轻微遮挡的图片：{"clarity": 0.65, "block": 0.2, "confidence": 0.75, "reasoning": "整体清晰但细节略模糊，机尾被建筑物轻微遮挡"}
- 对于模糊、明显遮挡的图片：{"clarity": 0.4, "block": 0.6, "confidence": 0.6, "reasoning": "图片模糊，约一半飞机被云层遮挡"}"""

        user_prompt = """请对这张飞机图片进行质量评估。

请根据清晰度和遮挡程度评分标准，给出客观的评估结果和置信度，并说明评估理由。"""

        return PromptResult(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            task_type="quality",
            metadata={}
        )

    def list_available_tasks(self) -> List[str]:
        """
        获取所有可用的任务类型

        Returns:
            List[str]: 任务类型列表
        """
        return list(self._prompts.keys())


# 全局Prompt引擎实例
prompt_engine = PromptEngine()


def get_prompt(task_type: str, **kwargs) -> PromptResult:
    """
    获取指定任务的prompt（便捷函数）

    Args:
        task_type: 任务类型
        **kwargs: 任务特定参数

    Returns:
        PromptResult: prompt结果
    """
    return prompt_engine.get_prompt(task_type, **kwargs)


def list_tasks() -> List[str]:
    """
    获取所有可用的任务类型（便捷函数）

    Returns:
        List[str]: 任务类型列表
    """
    return prompt_engine.list_available_tasks()
