"""
QAAgent - 知识库问答 agent

两种模式（由 settings.qa_agent_enabled 控制，默认 agent）：

    agent（默认）
        模型在工具循环里自己收集证据：可以多次检索、换措辞再搜、读整篇文档、
        跨文档查。收集够了再基于这些证据作答。
        —— 解决的是"固定单次检索"做不到的事：追问、通读、对比、搜不到重试。

    workflow（降级）
        问题 -> 向量召回 Top-K -> 构造上下文 -> LangChain Chain -> 答案
        agent 循环异常或超出预算时自动退化为这条路径，保证不会比原来更差。
"""
from typing import Any, AsyncIterator, Dict, List, Optional

from loguru import logger

from app.agents.base_agent import BaseAgent
from app.agents.qa_tools import TOOL_SCHEMAS, QAToolbox
from app.config import get_settings
from app.services.milvus_store import milvus_store

settings = get_settings()

# 从历史里截取多少轮，以及每轮答案保留多少字（历史只用于"理解在问什么"）
_HISTORY_TURNS = 4
_HISTORY_ANSWER_CHARS = 120

_SYSTEM_PROMPT = """你是「智能知识管理系统」的问答助手。请严格依据用户提供的【参考资料】回答问题。

回答要求：
1. 只使用参考资料中的信息，不要编造。资料不足时明确说明"资料中未提及"。
2. 用中文回答，条理清晰；涉及步骤或要点时使用列表。
3. 在引用具体内容处用 [1] [2] 这样的编号标注来源，编号对应参考资料序号。
4. 参考资料标题里标注了文档名（文件名）与标题。如果用户直接提到某个文档名，
   请优先依据那份文档作答，并在回答开头点明是哪份文档。
5. 如果提供了【对话历史】，结合上下文理解追问（例如"那第二点呢"），
   但事实依据仍然只能来自参考资料。
6. 不要重复罗列资料原文，要归纳总结。
"""

_USER_TEMPLATE = "{prompt}"

# ---------------- 证据收集循环的提示词 ----------------
_RESEARCH_SYSTEM = """你是知识库问答 agent。你的任务不是写答案，而是**收集足以回答问题的资料**。

可用工具：search_kb（语义检索）、search_by_filename（按名字找文档）、
read_document（读某篇文档的原文）、list_documents（列出所有文档）。

**停止条件（最重要）**：只要手上已有的片段能直接回答问题，立刻调用 finish。
不要为了"更稳妥""更全面"而多调工具——每多一步就多一次模型往返，用户要多等几秒。
一次 search_kb 就拿到明确答案的简单事实题，应当 1~2 步就 finish。

策略：
1. 如果给了【对话历史】，先判断当前问题到底在问什么。
   遇到"那第二点呢""这个呢""它"这类指代，必须先从历史里推断出完整问题，
   再用改写后的完整问题去检索——不要拿代词本身去搜。这一步是必须的。
2. 检索结果明显不相关（结论性内容缺失），才换一组关键词重搜；最多重搜 2 次，
   不要重复同一个 query。
3. 只有以下情况才用 read_document：用户要求通读/总结整篇，
   或检索命中的片段明显是"半句话"，需要看上下文。
4. 只有用户问"有哪些文档"或明确要做跨文档对比时，才用 list_documents。
5. 用户点名某份文档、但检索没覆盖到时，用 search_by_filename 定位。
6. 换了两三种方式确实找不到，才调用 give_up。
7. 不要编造，不要把猜测当成事实。"""

_LOOP_TOOLS = TOOL_SCHEMAS + [
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "已收集到足够回答问题的资料，结束收集。",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string", "description": "简短说明"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "give_up",
            "description": "换了多种方式检索仍找不到相关内容，结束收集。",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string", "description": "简短说明"}},
                "required": [],
            },
        },
    },
]


class QAAgent(BaseAgent):
    """知识库问答智能体"""

    name = "qa_agent"
    description = "基于知识库检索结果回答问题，并给出引用来源"

    # ---------- 提示词构建 ----------
    @staticmethod
    def _build_prompt(
        context: str,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """组装 human 消息：对话历史 + 参考资料 + 问题"""
        blocks: List[str] = []
        if history:
            lines = []
            for item in history:
                who = "用户" if item.get("role") == "user" else "助手"
                lines.append(f"{who}：{item.get('content', '')}")
            blocks.append("【对话历史】\n" + "\n".join(lines))
        blocks.append("【参考资料】\n" + context)
        blocks.append("【问题】\n" + question)
        blocks.append("请基于参考资料作答。")
        return "\n\n".join(blocks)

    @staticmethod
    def _build_context(chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return "（无相关资料）"
        blocks = []
        for i, c in enumerate(chunks, start=1):
            title = c.get("title") or "未知文档"
            filename = c.get("filename")
            # 文件名单独标出来：很多文档的标题是从正文首行抽的，
            # 和用户口中的"那个文件"对不上，把原名列出来模型才认得出
            source = f"《{title}》"
            if filename and filename != title:
                source += f"（文件名：{filename}）"
            blocks.append(
                f"[{i}] 来源：{source} 第{c.get('chunk_index', 0) + 1}片段\n"
                f"{c.get('content', '').strip()}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _citations(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        cites = []
        for i, c in enumerate(chunks, start=1):
            key = (c["document_id"], c["chunk_index"])
            if key in seen:
                continue
            seen.add(key)
            cites.append(
                {
                    "index": i,
                    "document_id": c["document_id"],
                    "title": c.get("title") or c.get("filename"),
                    "filename": c.get("filename"),
                    "chunk_index": c.get("chunk_index"),
                    "score": c.get("score"),
                    "vector_score": c.get("vector_score"),
                    "name_score": c.get("name_score", 0.0),
                    "match_type": c.get("match_type", "vector"),
                    "preview": (c.get("content") or "")[:200],
                }
            )
        return cites

    # ---------- 检索 ----------
    @staticmethod
    def _enrich(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """补充标题（文件名优先用 Milvus 里记录的原值）"""
        if not hits:
            return hits

        from app.db.database import session_scope
        from app.models.document import Document

        ids = {h["document_id"] for h in hits}
        with session_scope() as db:
            rows = db.query(Document).filter(Document.id.in_(ids)).all()
            meta_map = {
                r.id: {"title": r.title or r.filename, "filename": r.filename}
                for r in rows
            }

        for h in hits:
            meta = meta_map.get(h["document_id"], {})
            h["title"] = meta.get("title") or h.get("filename")
            h["filename"] = meta.get("filename") or h.get("filename")
        return hits

    def _retrieve(
        self,
        question: str,
        top_k: int,
        document_ids: Optional[List[int]],
        use_name_match: bool = True,
    ) -> List[Dict[str, Any]]:
        from app.services.retrieval_service import retrieve

        hits = retrieve(
            query=question,
            top_k=top_k,
            document_ids=document_ids,
            use_name_match=use_name_match,
        )["hits"]
        return self._enrich(hits)

    # ---------- 证据收集（agent 循环）----------
    @staticmethod
    def _history_text(history: Optional[List[Dict[str, str]]]) -> str:
        """把最近几轮对话压成一段提示。历史只用于理解"在问什么"，不提供事实。"""
        if not history:
            return ""
        lines = ["【对话历史】（仅用于理解当前问题在问什么，事实必须来自检索结果）"]
        for item in history[-_HISTORY_TURNS * 2 :]:
            who = "用户" if item.get("role") == "user" else "助手"
            content = (item.get("content") or "").strip().replace("\n", " ")
            if who == "助手":
                content = content[:_HISTORY_ANSWER_CHARS]
            lines.append(f"{who}：{content}")
        return "\n".join(lines) + "\n\n"

    def _make_toolbox(
        self,
        top_k: int,
        document_ids: Optional[List[int]],
        use_name_match: bool,
    ) -> QAToolbox:
        """创建工具箱（本地直调或 MCP 代理）"""
        transport = str(settings.qa_tool_transport).lower()
        if transport == "mcp":
            # MCP 模式：工具调用通过 stdio MCP server 中转
            # 需要先启动 backend/app/mcp_server.py
            from app.agents.qa_mcp_client import QAMCPToolbox
            return QAMCPToolbox(top_k, document_ids, use_name_match)
        # 默认 local：直接调用
        return QAToolbox(top_k, document_ids, use_name_match)

    def solve(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        top_k: Optional[int] = None,
        document_ids: Optional[List[int]] = None,
        use_name_match: bool = True,
    ) -> Dict[str, Any]:
        """
        运行证据收集循环

        Returns:
            {
              "chunks": 累积的片段（按首次出现顺序）,
              "steps": 每一步的动作轨迹,
              "gave_up": 模型是否主动放弃,
              "queries": 实际用过的检索语句（便于复盘与评测）,
            }
        """
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

        top_k = int(top_k or settings.search_top_k)
        talk = int(settings.qa_agent_max_steps)
        toolbox = self._make_toolbox(top_k, document_ids, use_name_match)

        llm = self.llm.bind_tools(_LOOP_TOOLS)
        messages = [
            SystemMessage(content=_RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"{self._history_text(history)}"
                    f"【当前问题】{question}\n\n"
                    "请开始收集资料，够了就调用 finish。"
                )
            ),
        ]

        steps: List[Dict[str, Any]] = []
        queries: List[str] = []
        gave_up = False
        finished = False

        try:
            for step in range(1, talk + 1):
                resp = llm.invoke(messages)
                messages.append(resp)
                calls = getattr(resp, "tool_calls", None) or []

                # 只思考没动手：已有证据就当作够了，否则推一把
                if not calls:
                    if toolbox.is_empty():
                        messages.append(HumanMessage(content="请先调用 search_kb 检索资料。"))
                        steps.append(
                            {"step": step, "tool": "thought", "args": {},
                             "observation": (resp.content or "")[:120]}
                        )
                        continue
                    steps.append(
                        {"step": step, "tool": "thought", "args": {},
                         "observation": (resp.content or "")[:120]}
                    )
                    finished = True
                    break

                for call in calls:
                    name = call.get("name") or ""
                    args = call.get("args") or {}
                    call_id = call.get("id") or f"call_{step}"

                    if name in ("finish", "give_up"):
                        gave_up = name == "give_up"
                        finished = True
                        steps.append(
                            {"step": step, "tool": name, "args": {},
                             "observation": str(args.get("reason") or "")[:120]}
                        )
                        break

                    observation = toolbox.call(name, args)
                    messages.append(ToolMessage(content=observation, tool_call_id=call_id))
                    if name == "search_kb":
                        queries.append(str(args.get("query") or ""))
                    steps.append(
                        {
                            "step": step,
                            "tool": name,
                            "args": args,
                            "observation": observation[:160],
                            "evidence": len(toolbox.evidence),
                            "chars": toolbox.chars,
                        }
                    )
                    if toolbox.chars > 16000:
                        messages.append(
                            HumanMessage(content="读取预算已用尽，请立即调用 finish 基于现有资料结束。")
                        )

                if finished:
                    break

            chunks = self._enrich(toolbox.collected())

            # 循环一个工具都没调（或全失败）：退回单次检索，保证有资料可用
            if not chunks:
                logger.info("agent 循环未收集到证据，退回单次检索")
                chunks = self._retrieve(question, top_k, document_ids, use_name_match)

            return {
                "chunks": chunks,
                "steps": steps,
                "gave_up": gave_up,
                "queries": queries,
                "chars": toolbox.chars,
            }
        finally:
            # MCP transport 持有常驻子进程与后台线程，必须释放
            close = getattr(toolbox, "close", None)
            if close:
                close()

    # ---------- 是否需要交给 agent ----------
    # 触发 agent 的词：都是"单次检索结构上做不到"的需求。
    # 刻意不含"比较"——"比较的损失函数是哪一个"是在问某个值，不是要做对比，
    # 把它算进来会让简单问题白白多花 4 秒。
    _TRIGGER_WORDS = (
        "总结", "概述", "全文", "通读", "对比", "区别", "哪个更好",
        "都有哪些", "有哪些", "列出", "分别有",
    )
    _REFER_WORDS = ("那", "这", "它", "他们", "其中", "上面", "刚才", "前面")

    def _needs_agent(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]],
    ) -> bool:
        """
        判断这个问题是否值得走工具循环

        闸门的意义：agent 每多一步就多一次模型往返（约 1.5~2s）。实测普通事实题
        走完循环要 8~10 秒，而单次检索 2 秒就能答对——不值得。
        所以只在"一次检索明显不够"的时候才升级成 agent：

            - 检索没命中 / 最高分偏低        -> 需要换措辞重搜
            - 存在对话上下文的短追问          -> 需要先把指代展开成完整问题
            - 问题本身要求通读、对比、列清单   -> 单次检索结构上就做不到
        """
        if not chunks:
            return True

        q = (question or "").strip()
        if any(word in q for word in self._TRIGGER_WORDS):
            return True

        if history and (len(q) <= 12 or any(q.startswith(w) for w in self._REFER_WORDS)):
            return True

        try:
            top = float(chunks[0].get("score") or 0.0)
        except (TypeError, ValueError):
            top = 0.0
        return top < float(settings.qa_agent_score_gate)

    # ---------- 证据收集的统一入口（带降级）----------
    def collect(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        top_k: Optional[int] = None,
        document_ids: Optional[List[int]] = None,
        use_name_match: bool = True,
        force_agent: bool = False,
    ) -> Dict[str, Any]:
        """
        收集回答所需的证据

        先用固定单次检索拿到"基础证据"，再判断要不要交给 agent 循环：
            - 不需要 -> 直接用单次结果作答（快，行为和改造前一致）
            - 需要   -> 进入工具循环，可用多轮检索/读整篇/跨文档
        agent 关闭或循环异常时同样退回单次结果——任何情况下都不该比改造前更差。
        """
        top_k = int(top_k or settings.search_top_k)
        base = self._retrieve(question, top_k, document_ids, use_name_match)

        if not settings.qa_agent_enabled and not force_agent:
            return {
                "chunks": base,
                "steps": [],
                "queries": [],
                "gave_up": False,
                "degraded": True,
            }

        if not force_agent and not self._needs_agent(question, base, history):
            return {
                "chunks": base,
                "steps": [],
                "queries": [],
                "gave_up": False,
                "gated": True,
            }

        try:
            return self.solve(
                question,
                history=history,
                top_k=top_k,
                document_ids=document_ids,
                use_name_match=use_name_match,
            )
        except Exception as exc:
            logger.warning(f"agent 循环失败，降级为单次检索: {exc}")
            return {
                "chunks": base,
                "steps": [],
                "queries": [],
                "gave_up": False,
                "degraded": True,
            }

    # ---------- 同步问答 ----------
    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Kwargs:
            question: 用户问题（必填）
            top_k: 召回条数
            document_ids: 限定文档范围
            history: 最近几轮对话 [{"role": "user"/"assistant", "content": str}]
        """
        question = (kwargs.get("question") or "").strip()
        if not question:
            raise ValueError("问题不能为空")

        top_k = int(kwargs.get("top_k") or settings.search_top_k)
        document_ids = kwargs.get("document_ids")
        use_name_match = kwargs.get("use_name_match", True)
        history = kwargs.get("history") or []

        solved = self.collect(question, history, top_k, document_ids, use_name_match)
        chunks = solved["chunks"]

        if not chunks:
            return {
                "question": question,
                "answer": "知识库中暂未检索到与该问题相关的内容。请先上传相关文档，或换一种问法试试。",
                "citations": [],
                "retrieved": 0,
                "steps": solved["steps"],
                "queries": solved["queries"],
            }

        context = self._build_context(chunks)
        answer = self._run_chain(question, context, history)

        return {
            "question": question,
            "answer": answer,
            "citations": self._citations(chunks),
            "retrieved": len(chunks),
            "steps": solved["steps"],
            "queries": solved["queries"],
            "gave_up": solved.get("gave_up", False),
            "degraded": solved.get("degraded", False),
        }

    def _run_chain(
        self,
        question: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """LangChain LCEL 链：Prompt | LLM | StrOutputParser"""
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT),
                ("human", _USER_TEMPLATE),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {"prompt": self._build_prompt(context, question, history)}
        ).strip()

    # ---------- 流式问答 ----------
    async def astream(self, **kwargs: Any) -> AsyncIterator[str]:
        """SSE 流式输出（逐段 yield 文本）"""
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        question = (kwargs.get("question") or "").strip()
        if not question:
            yield "问题不能为空"
            return

        top_k = int(kwargs.get("top_k") or settings.search_top_k)
        document_ids = kwargs.get("document_ids")
        use_name_match = kwargs.get("use_name_match", True)
        history = kwargs.get("history") or []
        # 流式接口会先调 prepare_stream 拿引用，把检索结果传进来避免重复检索
        # （重复一次 = 多一次 BGE 编码 + 多一次 Milvus 查询）
        chunks = kwargs.get("chunks")
        if chunks is None:
            chunks = self.collect(
                question, history, top_k, document_ids, use_name_match
            )["chunks"]

        if not chunks:
            yield "知识库中暂未检索到与该问题相关的内容。请先上传相关文档，或换一种问法试试。"
            return

        context = self._build_context(chunks)
        prompt = ChatPromptTemplate.from_messages(
            [("system", _SYSTEM_PROMPT), ("human", _USER_TEMPLATE)]
        )
        chain = prompt | self.llm | StrOutputParser()

        payload = {"prompt": self._build_prompt(context, question, history)}
        async for piece in chain.astream(payload):
            if piece:
                yield piece

    def prepare_stream(self, **kwargs: Any) -> Dict[str, Any]:
        """SSE 场景下先跑证据收集，返回引用与轨迹，再由 astream 流式作答"""
        question = (kwargs.get("question") or "").strip()
        solved = self.collect(
            question,
            kwargs.get("history") or [],
            kwargs.get("top_k"),
            kwargs.get("document_ids"),
            kwargs.get("use_name_match", True),
        )
        chunks = solved["chunks"]
        return {
            "chunks": chunks,
            "citations": self._citations(chunks) if chunks else [],
            "steps": solved["steps"],
            "queries": solved["queries"],
            "degraded": solved.get("degraded", False),
        }

    def health(self) -> Dict[str, Any]:
        ok = milvus_store.health()
        return {"agent": self.name, "milvus": ok, "status": "ok" if ok else "down"}


qa_agent = QAAgent()
