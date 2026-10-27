"""
DeepSeek LLM 服务

DeepSeek 提供兼容 OpenAI 的接口，直接使用 openai SDK 即可。
"""
from typing import Dict, Iterator, List, Optional

from loguru import logger
from openai import OpenAI

from app.config import get_settings

settings = get_settings()


class LLMService:
    """DeepSeek 对话服务（延迟初始化）"""

    _instance: "LLMService | None" = None
    _client: Optional[OpenAI] = None

    def __new__(cls) -> "LLMService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if not settings.deepseek_api_key:
                raise RuntimeError("未配置 DEEPSEEK_API_KEY")
            self._client = OpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                timeout=60.0,
                max_retries=2,
            )
            logger.info(f"DeepSeek 客户端就绪 | 模型={settings.deepseek_model}")
        return self._client

    # ---------- 基础对话 ----------
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        response = self.client.chat.completions.create(
            model=settings.deepseek_model,
            messages=messages,
            temperature=(
                settings.llm_temperature if temperature is None else temperature
            ),
            max_tokens=settings.llm_max_tokens if max_tokens is None else max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    def simple_chat(self, prompt: str, system: Optional[str] = None) -> str:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages)

    # ---------- 流式对话 ----------
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=settings.deepseek_model,
            messages=messages,
            temperature=(
                settings.llm_temperature if temperature is None else temperature
            ),
            max_tokens=settings.llm_max_tokens,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                yield piece

    # ---------- 健康检查 ----------
    def health(self) -> bool:
        try:
            # 注意：deepseek-flash 是推理模型，会先输出 reasoning_content 再输出正文。
            # max_tokens 给太小（比如 16）会被思考过程吃光，content 返回空字符串，
            # 看起来像"成功但没内容"。所以这里给足预算，并且把空回复判为失败。
            reply = self.chat(
                [{"role": "user", "content": "回复两个字：正常"}],
                max_tokens=128,
                temperature=0.0,
            )
            if not reply:
                logger.error("DeepSeek 健康检查失败：返回内容为空（可能是 token 预算不足）")
                return False
            logger.info(f"DeepSeek 健康检查: {reply}")
            return True
        except Exception as exc:
            logger.error(f"DeepSeek 健康检查失败: {exc}")
            return False


llm_service = LLMService()
