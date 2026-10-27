"""
SummaryAgent - 文档摘要与关键词（ReAct 工具循环）

为什么这里需要 agent：
    原来的实现是"取正文前 2500 字，一次性让模型写摘要"。实测本机 6 篇文档里，
    有 3 篇的字数在 5900~7600 之间，也就是说超过一半的内容摘要根本没看到。
    这不是参数没调好，是"截断"这个做法本身的信息损失。

    摘要任务恰好具备 ReAct 成立的两个前提：
        1. 信息不在起始位置——结论、数据往往在文档后半部分
        2. 读哪里需要判断——文档类型杂（简历、问卷、技术方案），
           没有任何固定规则能决定该读哪几段

    所以这里让模型自己看结构、自己决定读哪段、不够再读。问答链路没有这个需要
    （检索工具的选择几乎确定），因此仍然是固定工作流，没有被改成 agent。

两种模式：
    react  : 文档长度超过阈值时启用，模型在工具循环里自己收集信息
    single : 短文档直接一次调用（全文都在提示词里，没有"读哪里"的问题）

预算与兜底（缺一不可，否则会卡死在 processing）：
    max_steps        循环步数上限
    max_read_chunks  单次最多读几块，防止一次性灌满上下文
    max_total_chars  累计读入上限，超了强制收尾
    读完一轮仍没拿到结果 → 用已读内容做一次 reduce，保证一定有输出
"""
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from app.agents.base_agent import BaseAgent
from app.db.database import session_scope
from app.models.document import Document

_SUMMARY_SYSTEM = (
    "你是一名专业的中文文档分析助手。请严格按用户要求输出，不要添加多余的解释。"
)

_JSON_RE = re.compile(r"\{.*\}", re.S)

# ---------------- ReAct 提示词 ----------------
_REACT_SYSTEM = """你是一个文档摘要 agent，任务是为一篇文档生成摘要与关键词。

你可以调用工具：查看结构、读取原文片段、在文档内检索。

工作方式：
1. 先调用 get_outline 了解文档结构，不要一上来就读全文。
2. 根据结构判断哪些部分最可能包含主题与结论，再用 read_chunks 按需读取。
3. 已经读过的片段不要重复读；也不要为了凑步数而读。
4. 涉及数字、结论、专有名词时，务必确认它们来自你真正读过的原文，不要凭印象编造。
5. 收集到足够信息后，必须调用 finish 提交结果。

finish 的约束：summary 用中文、150 字以内，说清文档主题与核心结论；
keywords 给 3~5 个，按重要性排序。"""

# ---------------- 工具定义 ----------------
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_outline",
            "description": "查看文档结构：分块总数，以及每个分块开头的文字。应先调用它。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_chunks",
            "description": "读取指定范围的分块原文，下标从 0 开始、闭区间。单次最多读 6 块。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "integer", "description": "起始分块下标"},
                    "end": {"type": "integer", "description": "结束分块下标（含）"},
                },
                "required": ["start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_document",
            "description": "在本文档内做关键词检索，返回最相关的分块，适合先定位'结论''数据''结果'这类内容。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "检索关键词"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "提交最终摘要与关键词。调用后任务结束。",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "150 字以内中文摘要"},
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "3~5 个关键词",
                    },
                },
                "required": ["summary", "keywords"],
            },
        },
    },
]


class SummaryAgent(BaseAgent):
    """文档摘要智能体：短文档单次生成，长文档走 ReAct 工具循环"""

    name = "summary_agent"
    description = "为文档生成中文摘要与关键词（长文档先看结构再按需读原文）"

    # ---------- 预算 ----------
    max_steps = 6
    max_read_chunks = 6
    max_total_chars = 20000
    # 低于该字符数直接单次生成：全文都在提示词里，"读哪里"不是问题
    short_doc_chars = 3500

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Kwargs:
            text: 正文（与 document_id 二选一）
            document_id: 从数据库读取分块
            filename: 文件名，用于提示
            persist: 是否写回 MySQL，默认 False
            force_react: 强制走工具循环（测试用）
        """
        text = kwargs.get("text")
        document_id = kwargs.get("document_id")
        filename = kwargs.get("filename", "") or ""
        persist = kwargs.get("persist", False)
        force_react = bool(kwargs.get("force_react", False))

        chunks = self._load_chunks(document_id) if document_id else []
        body = text or "\n".join(c["content"] for c in chunks)
        if not body.strip():
            raise ValueError("缺少可用文本")

        # 分块读不到时只能单次摘要——这条路径会丢信息，必须让它可见
        if not chunks and document_id and len(body) > self.short_doc_chars:
            logger.warning(
                f"文档 {document_id} 未读到分块（正文 {len(body)} 字），"
                f"退化为单次摘要，仅覆盖前 8000 字"
            )

        total_chars = sum(len(c["content"]) for c in chunks) or len(body)
        use_react = bool(chunks) and (
            force_react
            or len(chunks) > self.max_read_chunks
            or total_chars > self.short_doc_chars
        )

        if use_react:
            summary, keywords, trace = self._react(filename, chunks)
            mode = "react"
        else:
            summary, keywords = self._single_shot(body, filename)
            trace = []
            mode = "single"

        if persist and document_id:
            self._persist(document_id, summary, keywords, trace)

        logger.info(
            f"摘要完成 | 文档={document_id or '-'} 模式={mode} "
            f"步数={len(trace)} 摘要长度={len(summary)} 关键词={len(keywords)}"
        )
        return {
            "summary": summary,
            "keywords": keywords,
            "trace": trace,
            "mode": mode,
        }

    # ---------- 模式一：单次生成 ----------
    def _single_shot(self, body: str, filename: str) -> Tuple[str, List[str]]:
        excerpt = body[:8000]
        prompt = (
            f"文档名称：{filename}\n"
            "请阅读以下文档内容，输出一个 JSON 对象，不要输出任何其他内容：\n"
            "{\n"
            '  "summary": "150字以内的中文摘要，说明文档主题与核心结论",\n'
            '  "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"]\n'
            "}\n\n"
            f"文档内容：\n{excerpt}"
        )
        raw = self.invoke_llm(prompt, system=_SUMMARY_SYSTEM, max_tokens=400)
        return self._parse(raw)

    # ---------- 模式二：ReAct 工具循环 ----------
    def _react(
        self, filename: str, chunks: List[Dict[str, Any]]
    ) -> Tuple[str, List[str], List[Dict[str, Any]]]:
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

        llm = self.llm.bind_tools(_TOOLS)
        messages = [
            SystemMessage(content=_REACT_SYSTEM),
            HumanMessage(
                content=(
                    f"文档名称：{filename or '未命名'}\n"
                    f"分块总数：{len(chunks)}\n"
                    "请开始分析，完成后调用 finish。"
                )
            ),
        ]

        trace: List[Dict[str, Any]] = []
        state: Dict[str, Any] = {"read": set(), "chars": 0, "seen": []}
        result: Optional[Tuple[str, List[str]]] = None

        for step in range(1, self.max_steps + 1):
            resp = llm.invoke(messages)
            messages.append(resp)
            calls = getattr(resp, "tool_calls", None) or []

            # 只思考没动手：推它一把，避免空转把步数耗完
            if not calls:
                thought = (resp.content or "").strip()
                trace.append(
                    {
                        "step": step,
                        "action": "thought",
                        "observation": thought[:120],
                        "chars": state["chars"],
                    }
                )
                nudge = (
                    "请先调用 get_outline 了解文档结构。"
                    if state["chars"] == 0
                    else "如果信息已足够请调用 finish；不够请继续读。"
                )
                messages.append(HumanMessage(content=nudge))
                continue

            finished = False
            for call in calls:
                name = call.get("name") or ""
                args = call.get("args") or {}
                call_id = call.get("id") or f"call_{step}"

                if name == "finish":
                    summary = str(args.get("summary") or "").strip()
                    keywords = [
                        str(k).strip() for k in (args.get("keywords") or []) if str(k).strip()
                    ]
                    trace.append(
                        {
                            "step": step,
                            "action": "finish",
                            "observation": summary[:120],
                            "chars": state["chars"],
                        }
                    )
                    if not summary:
                        messages.append(
                            ToolMessage(content="summary 不能为空，请重新调用 finish。", tool_call_id=call_id)
                        )
                        continue
                    result = (summary, keywords[:8])
                    finished = True
                    break

                observation = self._execute(name, args, chunks, state)
                messages.append(ToolMessage(content=observation, tool_call_id=call_id))
                trace.append(
                    {
                        "step": step,
                        "action": name,
                        "args": args,
                        "observation": observation[:120],
                        "chars": state["chars"],
                    }
                )

            if finished:
                break

            if state["chars"] > self.max_total_chars:
                messages.append(
                    HumanMessage(content="已超出读取预算，请立即调用 finish 提交结果。")
                )

        if result is None:
            # 兜底：步数用完或预算超了，用已读内容强制收尾
            logger.warning("ReAct 未在预算内产出结果，转为 reduce 兜底")
            body = "\n\n".join(item["text"] for item in state["seen"])[:8000]
            if not body:
                body = "\n".join(c["content"] for c in chunks)[:8000]
            summary, keywords = self._single_shot(body, filename)
            trace.append(
                {
                    "step": len(trace) + 1,
                    "action": "reduce_fallback",
                    "observation": f"用已读 {state['chars']} 字收尾",
                    "chars": state["chars"],
                }
            )
            result = (summary, keywords)

        return result[0], result[1], trace

    # ---------- 工具执行 ----------
    def _execute(
        self,
        name: str,
        args: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        state: Dict[str, Any],
    ) -> str:
        try:
            if name == "get_outline":
                return self._tool_outline(chunks)
            if name == "read_chunks":
                return self._tool_read(args, chunks, state)
            if name == "search_in_document":
                return self._tool_search(args, chunks, state)
            return f"未知工具：{name}。可用工具：get_outline / read_chunks / search_in_document / finish"
        except Exception as exc:  # 工具出错不该中断循环，交回模型自己处理
            logger.warning(f"摘要工具 {name} 执行失败: {exc}")
            return f"工具执行失败：{exc}"

    @staticmethod
    def _tool_outline(chunks: List[Dict[str, Any]]) -> str:
        lines = [f"共 {len(chunks)} 个分块（下标 0~{len(chunks) - 1}）："]
        for c in chunks:
            head = re.sub(r"\s+", " ", c["content"].strip())[:60]
            lines.append(f"[{c['index']}] {head}")
        return "\n".join(lines)

    def _tool_read(
        self, args: Dict[str, Any], chunks: List[Dict[str, Any]], state: Dict[str, Any]
    ) -> str:
        try:
            start = int(args.get("start", 0))
            end = int(args.get("end", start))
        except (TypeError, ValueError):
            return "start / end 必须是整数"
        if end < start:
            start, end = end, start
        start = max(0, start)
        end = min(len(chunks) - 1, end)
        if start > end:
            return f"下标超出范围（0~{len(chunks) - 1}）"
        if end - start + 1 > self.max_read_chunks:
            end = start + self.max_read_chunks - 1
            note = f"（单次最多 {self.max_read_chunks} 块，已截到 {start}~{end}）\n"
        else:
            note = ""

        key = (start, end)
        if key in state["read"]:
            return f"分块 {start}~{end} 已经读过了。请读别的部分，或直接调用 finish。"

        state["read"].add(key)
        picked = chunks[start : end + 1]
        text = "\n\n".join(f"【分块{c['index']}】\n{c['content']}" for c in picked)
        state["chars"] += len(text)
        state["seen"].append({"range": key, "text": text})
        return note + text

    def _tool_search(
        self, args: Dict[str, Any], chunks: List[Dict[str, Any]], state: Dict[str, Any]
    ) -> str:
        from app.services.bm25 import BM25Index
        from app.utils.chinese_tokenizer import tokenize

        query = str(args.get("query") or "").strip()
        if not query:
            return "query 不能为空"

        corpus = [list(tokenize(c["content"])) for c in chunks]
        index = BM25Index(corpus)
        scores = index.scores(list(tokenize(query)))

        ranked = sorted(range(len(chunks)), key=lambda i: -float(scores[i]))[:3]
        ranked = [i for i in ranked if float(scores[i]) > 0]
        if not ranked:
            return f"文档中未找到与「{query}」相关的内容。"

        blocks = []
        for i in ranked:
            content = chunks[i]["content"]
            pos = content.find(query)
            if pos < 0:
                pos = 0
            snippet = re.sub(r"\s+", " ", content[max(0, pos - 60) : pos + 200]).strip()
            blocks.append(f"[{chunks[i]['index']}] 命中「{query}」附近：{snippet}")

        state["seen"].append({"range": ("search", query), "text": "\n".join(blocks)})
        return "\n".join(blocks)

    # ---------- 内部方法 ----------
    def _load_chunks(self, document_id: int) -> List[Dict[str, Any]]:
        from app.models.document import Chunk

        with session_scope() as db:
            rows = (
                db.query(Chunk)
                .filter(Chunk.document_id == document_id)
                .order_by(Chunk.chunk_index)
                .all()
            )
        return [
            {"index": i, "content": r.content}
            for i, r in enumerate(rows)
        ]

    def _persist(
        self,
        document_id: int,
        summary: str,
        keywords: List[str],
        trace: List[Dict[str, Any]],
    ) -> None:
        with session_scope() as db:
            doc = db.get(Document, document_id)
            if doc is None:
                return
            doc.summary = summary
            doc.keywords = json.dumps(keywords, ensure_ascii=False)
            if trace:
                doc.summary_trace = json.dumps(trace, ensure_ascii=False)
        logger.info(f"文档 {document_id} 摘要已写入")

    @staticmethod
    def _parse(raw: str) -> Tuple[str, List[str]]:
        """容错解析 LLM 返回的 JSON"""
        match = _JSON_RE.search(raw or "")
        if match:
            try:
                data = json.loads(match.group(0))
                summary = str(data.get("summary", "")).strip()
                keywords = data.get("keywords") or []
                if isinstance(keywords, str):
                    keywords = [k.strip() for k in re.split(r"[,，;；]", keywords) if k.strip()]
                return summary, [str(k) for k in keywords][:12]
            except json.JSONDecodeError:
                pass
        # 解析失败则退化为纯文本
        return (raw or "").strip()[:600], []

    def health(self) -> Dict[str, Any]:
        return {
            "agent": self.name,
            "status": "ok",
            "max_steps": self.max_steps,
            "short_doc_chars": self.short_doc_chars,
            "max_total_chars": self.max_total_chars,
        }


summary_agent = SummaryAgent()
