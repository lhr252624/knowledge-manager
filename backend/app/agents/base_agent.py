"""
Agent 基类

所有智能体统一继承 BaseAgent，约定：
    - name / description: 用于编排层展示与路由
    - run(payload): 执行主体逻辑，返回 dict
    - health(): 自检

LLM 通过 LangChain 的 ChatOpenAI 接入 DeepSeek（OpenAI 兼容协议）。
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from loguru import logger

from app.config import get_settings

settings = get_settings()


class BaseAgent(ABC):
    """智能体基类"""

    name: str = "base_agent"
    description: str = "基础智能体"

    def __init__(self) -> None:
        self._llm = None

    # ---------- LLM ----------
    @property
    def llm(self):
        """LangChain ChatModel（延迟构建）"""
        if self._llm is None:
            from langchain_openai import ChatOpenAI

            if not settings.deepseek_api_key:
                raise RuntimeError("未配置 DEEPSEEK_API_KEY，无法使用 LLM")
            self._llm = ChatOpenAI(
                model=settings.deepseek_model,
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                timeout=90,
                max_retries=2,
            )
            logger.debug(f"[{self.name}] LangChain ChatOpenAI 已就绪")
        return self._llm

    # ---------- 约定接口 ----------
    @abstractmethod
    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """执行智能体任务"""

    def health(self) -> Dict[str, Any]:
        """自检"""
        return {"agent": self.name, "status": "unknown"}

    # ---------- 工具方法 ----------
    def invoke_llm(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """同步调用 LLM 返回纯文本"""
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))

        # 按用途限制输出长度：摘要这类任务不需要 2000 tokens，
        # 限制后能显著缩短响应时间（DeepSeek 按 token 生成，输出越短越快）
        llm = self.llm.bind(max_tokens=max_tokens) if max_tokens else self.llm
        response = llm.invoke(messages)
        content = response.content
        if isinstance(content, list):  # 多模态返回时拼接文本片段
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content).strip()

    async def ainvoke_llm(self, prompt: str, system: Optional[str] = None) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))
        response = await self.llm.ainvoke(messages)
        return str(response.content).strip()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"
