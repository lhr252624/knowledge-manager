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
import json
import re
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

【最高优先级：形式服从请求】
用户怎么要求形式，就怎么输出；用户的形式要求优先于下面所有默认格式。
- 要"关键词"：只输出关键词本身，用「、」分隔，数量按用户说的（没说就给 3~5 个）；
  不分类、不编号、不带金额与数字、不解释、不加结尾说明。
- 要"一句话 / 简短 / 只要结论"：一到两句话，不超过 60 字。
- 要"详细 / 展开 / 分步骤"：才使用列表，允许多段。
- 没有指定形式：默认 200 字以内，先给结论再给依据。

【资料一致性】
落笔之前先判断：【参考资料】是否都在回答同一个问题。
- 如果它们分属不同主题（例如一份讲概念、一份讲另一个项目），只采用与问题最相关的那一份，
  其余不要采用，也不要在回答里提它们；更不要把不同主题的资料拼成一篇"大全"。
- 如果资料只是零散提到、并没有系统介绍，就只回答能被支撑的部分，并明确说明
  "知识库中没有关于 X 的系统介绍"，不要据此扩写成教科书式的长文。

【结构】
- 沿用资料本身的顺序和标题；不要为了显得完整而自造"一、二、三……"。
- 每一节都必须带来新信息；两节讲同一件事就合并成一节。
- 每一段只讲一个论点，段与段之间要有推进关系（是什么 → 怎么做 → 效果/边界），
  不要用"另外""此外"把不相关的内容并列。
- 引用标注挂在关键论断后面即可，一段最多一处，不要每句话都挂编号。

【事实纪律】
1. 只使用参考资料中的信息，不要编造。确实没有依据时用一句话说明"资料中未提及"，
   不要逐条罗列缺了哪些内容。
2. 在引用具体内容处用 [1] [2] 这样的编号标注来源，编号对应参考资料序号；
   纯列举关键词时不需要标注。
3. 参考资料标题里标注了文档名（文件名）与标题。用户点名了某份文档就直接作答，
   不要复述文档名；没点名且涉及多篇时才说明依据哪一份。
4. 如果提供了【对话历史】，结合上下文理解追问（例如"那第二点呢"），
   但事实依据仍然只能来自参考资料。
5. 不要重复罗列资料原文，要归纳。
"""

_USER_TEMPLATE = "{prompt}"

# ---------------- 输出形式识别 ----------------
# 只改提示词还不够：提示词里写"要简短"，模型仍然会写到 max_tokens 为止。
# 所以这里把形式识别出来，既写进提示词，也用来卡生成长度（见 _FORMAT_MAX_TOKENS）。
_FORMAT_KEYWORDS = re.compile(r"关键词|关键字|标签|几个词|提炼.{0,4}词")
_FORMAT_SHORT = re.compile(
    r"只要?结论|一句话|一两句|简短|简要|简单说|别展开|不要解释|不要正文|不需要解释|只给"
)
_FORMAT_DETAILED = re.compile(r"详细|展开|具体说明|分步骤|一步一步|完整说明|尽量全面")

# 顺序有意义：先说"关键词"再说"简短的"，"只给我 5 个关键词"要落到 keywords
_FORMAT_RULES = (
    ("keywords", _FORMAT_KEYWORDS),
    ("short", _FORMAT_SHORT),
    ("detailed", _FORMAT_DETAILED),
)

# 各形式允许的输出上限（token）。中文约 1.5 字/token，全局默认 2048 太大了；
# default 留 900 是给"对比/列清单"这类不指定形式但要展开的问题留余量，别把答案截断在半句
_FORMAT_MAX_TOKENS = {"keywords": 120, "short": 200, "detailed": 1024, "default": 900}

_FORMAT_HINTS = {
    "keywords": "【输出形式】只输出关键词，用「、」分隔，3~5 个；不解释、不编号、不分段。",
    "short": "【输出形式】一到两句话，60 字以内，不要列表。",
    "detailed": "【输出形式】可以分点展开，但不要重复罗列原文。",
    "default": "",
}


def detect_format(question: str) -> str:
    """识别用户要的输出形式：keywords / short / detailed / default（纯规则，零成本）"""
    q = (question or "").strip()
    for name, pattern in _FORMAT_RULES:
        if pattern.search(q):
            return name
    return "default"


# ---------------- 意图分诊 ----------------
# 寒暄、身份、能力这类问题没有"知识库内容"可查，走检索只会把无关片段翻出来，
# 再用 20 秒写一篇解释。这里用纯规则先分诊（零成本），命中就直接回固定话术。
_INTENT_RULES = (
    ("bye", re.compile(r"^(再见|拜拜|bye|晚安|先这样|下次聊|结束吧)[\s，,。.!！?？~]*$", re.I)),
    ("thanks", re.compile(r"^(谢谢|多谢|感谢|谢了|辛苦(了|啦)?|thanks|thank you|thx)(您|你|啦|了)?[\s，,。.!！?？~]*$", re.I)),
    ("identity", re.compile(r"(你是谁|你叫什么|你的名字|你是什么(人|东西|角色)?|介绍一下你(自己)?|自我介绍|你是(机器人|助手|AI|程序)吗)")),
    ("capability", re.compile(r"(你能做(什么|啥)|你会(做)?什么|能帮我做(什么|啥)|有什么(功能|能力)|怎么用你|你可以干(什么|啥)|你能干什么)")),
    ("greeting", re.compile(r"^(hi|hello|hey|嗨|哈喽|哈啰|你好|您好|在吗|在么|早|早上好|中午好|下午好|晚上好|早安)[\s，,。.!！?？~]*$", re.I)),
)
# 寒暄类只认"整句就是寒暄"，避免"你好，帮我看下这篇"被误判
_INTENT_MAX_LEN = 12

_SOCIAL_REPLY = {
    "greeting": (
        "你好！我是这套知识库的问答助手。直接把问题说给我就行——"
        "比如「某份文档讲了什么」「某个流程怎么做」「两份方案怎么选」。"
        "我会在你的资料里检索后作答，并标出来源。"
    ),
    "identity": (
        "我是这套知识库的问答助手，不是资料里的某个人。"
        "你提问，我在你上传的文档里检索，按资料作答并给出来源编号；"
        "资料里没有的，我会直接说「知识库里没有」，不会编。"
    ),
    "capability": (
        "我能做这几件事：按问题检索知识库并作答（带来源）；对比、总结几份文档；"
        "按你指定的形式输出（关键词、结论、分点）。"
        "检索只用已上传的资料，不查外部知识——所以先确认相关文档已入库。"
    ),
    "thanks": "不客气。还有问题随时问。",
    "bye": "好的，回头见。",
}


def detect_intent(question: str) -> str:
    """
    分诊：greeting / identity / capability / thanks / bye，普通问题返回 ""

    只做一件自己不会做错的事——把"没有知识库内容可查"的输入挡在检索之外。
    看不懂的（长句、带具体指代）一律交回正常链路，宁可多花时间也不误判。
    """
    q = (question or "").strip()
    if not q or len(q) > _INTENT_MAX_LEN:
        return ""
    for name, pattern in _INTENT_RULES:
        if pattern.search(q):
            return name

    # 组合寒暄："hi，你好" / "你好，在吗" / "嗨 早" —— 每段都是问候才算
    parts = [p for p in re.split(r"[\s，,。.!！?？~、]+", q) if p]
    greeting = dict(_INTENT_RULES)["greeting"]
    fillers = {"呀", "啊", "呢", "哈", "哦", "喔"}
    if len(parts) >= 2 and all(greeting.search(p) or p in fillers for p in parts):
        return "greeting"
    return ""


def social_reply(intent: str) -> str:
    return _SOCIAL_REPLY.get(intent, "")


# ---------------- 上下文组装 ----------------
# 一次回答最多采用几篇文档、每篇最多几段
_CTX_MAX_DOCS = 3
_CTX_MAX_CHUNKS_PER_DOC = 4
# 每个文档最多补几个相邻片段
_CTX_NEIGHBORS_PER_DOC = 3


def order_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    把检索片段按"文档聚簇 + 文档内连续窗口"重排

    检索结果是按分数排序的，同一篇文档的几个片段会被打散成 8/12/9/2 这种顺序，
    提示词里也就按这个顺序给模型——模型据此重建叙事时必然前后错位，
    这就是"看似答了很多、实际前言不搭后语"的直接原因。

    这里改成：先按文档分组（组间按该文档的最高分降序）；组内以最高分片段为锚点，
    向两侧优先取**连续**片段（凑成一段能读的原文），而不是再按分数跳着挑。
    连续比"每条都高分"更重要：碎片读起来就是段落抄一半。
    """
    if not chunks:
        return []

    groups: Dict[int, List[Dict[str, Any]]] = {}
    for c in chunks:
        groups.setdefault(int(c["document_id"]), []).append(c)

    best = lambda items: max(float(c.get("score") or 0) for c in items)  # noqa: E731
    ranked = sorted(groups.items(), key=lambda kv: -best(kv[1]))

    ordered: List[Dict[str, Any]] = []
    for _, items in ranked[:_CTX_MAX_DOCS]:
        by_idx: Dict[int, Dict[str, Any]] = {}
        for c in items:
            idx = int(c.get("chunk_index") or 0)
            # 同一块可能有两条（向量/名称两路召回），留分高的
            if idx not in by_idx or float(c.get("score") or 0) > float(
                by_idx[idx].get("score") or 0
            ):
                by_idx[idx] = c

        anchor = max(by_idx, key=lambda i: float(by_idx[i].get("score") or 0))
        window = [anchor]
        # 先往后扩（命中块通常只是开头，结论在后面），再往前补
        for step in range(1, _CTX_MAX_CHUNKS_PER_DOC):
            if anchor + step in by_idx:
                window.append(anchor + step)
            if len(window) >= _CTX_MAX_CHUNKS_PER_DOC:
                break
            if anchor - step in by_idx:
                window.append(anchor - step)
            if len(window) >= _CTX_MAX_CHUNKS_PER_DOC:
                break

        ordered.extend(by_idx[i] for i in sorted(window))
    return ordered


# ---------------- 作答提纲 ----------------
# "每一段都合理、放在一起没逻辑"的根因是：模型是逐段看资料、逐段写，
# 从头到尾没有定过"这一篇要论证什么"。所以先让它把骨架写出来，再按骨架填肉。
_OUTLINE_SYSTEM = """你是答题规划员。你的任务不是写答案，而是先定一个能自洽的提纲。

输出 3~5 行，每行格式：
<序号>. <这一节要讲什么，必须是完整的一句话> | 依据: [资料编号,...]

要求：
- 第一行先给结论或定义，直接回答"问题到底在问什么"。
- 后面几条要有推进关系（是什么 → 怎么做 → 效果/边界），不要照抄资料的小标题平铺。
- 每条都要能落到具体资料编号；落不到依据的就不要写进提纲。
- 只输出提纲，不要写正文，不要解释。"""

_OUTLINE_TEMPLATE = """【参考资料】
{context}

【问题】
{question}

请给出提纲。"""

_OUTLINE_HINT = """【作答提纲】（结构已定，请按它写）
{outline}

写作要求：一节一段、顺序不变；每节只讲这一件事；节与节之间要有衔接
（递进、因果、转折），不要用「另外」「此外」把不相关的内容并列。
提纲里某节如果资料支撑不足，就写「资料中未提及」，不要拿别的资料来填。"""


# ---------------- 生成后自检 ----------------
_SENT_SPLIT = re.compile(r"[。！？!?\n]+")

_REPAIR_SYSTEM = """你是答案校对员。给你一个问题、一份答案、若干资料片段，以及校对员发现的问题。
你的任务不是重写，而是**只修掉被指出的问题**，保留答案里正确、有依据的内容。

铁律：
- 不得引入资料里没有的新事实、新数字、新结论。
- 不要换行文风格，不要扩写，不要加总结段。
- 修不了（比如资料确实不足）就在对应位置改成"资料中未提及"，不要编。
- 只输出修正后的答案正文，不要解释你改了什么。"""

_REPAIR_TEMPLATE = """【问题】
{question}

【待校对答案】
{answer}

【校对员发现的问题】
{issues}

【资料片段（只能依据这些）】
{context}

请输出修正后的答案。"""


# ---------------- 检索规划 ----------------
# 同一个问题，检索方式不一样，召回的可能是完全不同的东西：
#   问"详细讲讲 agent"——直接检索会把知识库里唯一的"简历审查 Agent"捞上来，
#   于是回答全是一个具体项目的细节，用户问的却是通用概念。
# 所以先让模型规划：换措辞(HyDE)、拆子问题(decompose)，还是直接查，以及
# 这个问题到底在问"通用概念"还是"某个具体项目"。
_QUERY_PLAN_SYSTEM = """你是检索规划员。给定用户问题（可能附带对话历史），你决定怎么在知识库里检索。

三种策略：
- direct：问题指向明确，直接拿问题去检索。
- hyde：先写一段"假设的答案"，再拿这段答案去检索。
        适合问的是通用概念/抽象做法，而知识库里很可能用另一套说法描述同一件事。
- decompose：把问题拆成 2~4 个子问题，分别检索再合并。
        适合一句话里有多个子目标、或需要对比的问题。

同时判断问题的抽象层级：
- kb：问题指向知识库里已有的具体项目或文档。
- general：问的是通用概念/常识，知识库里可能只有一个具体实例，甚至没有。
- mixed：既问通用概念，又要落到具体项目。

只输出 JSON，不要解释，不要 markdown 代码块：
{"strategy":"direct|hyde|decompose","sub_queries":["..."],"scope":"kb|general|mixed","reason":"一句话"}"""

_QUERY_PLAN_TEMPLATE = """{history}【问题】{question}

请输出检索计划 JSON。"""

_HYDE_SYSTEM = "你是检索助手。你的任务不是回答用户，而是写一段用于检索的'假设答案'。"
_HYDE_TEMPLATE = """针对下面的问题，用你自己的知识写一段 150 字以内的、像样的回答。
不需要准确，也不要引用任何资料——它只用来当检索的语义桥梁。

问题：{question}"""

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)

_SCOPE_HINTS = {
    "general": (
        "【知识边界】用户问的是通用概念，而知识库里主要是具体项目文档。请分两层作答：\n"
        "① 第一层「通用说明」：用你自己的知识把这个概念讲完整——是什么、由哪几部分组成、"
        "有哪几种常见范式、按什么维度分类、通常用在什么场景。这一层不要依赖某个具体框架或项目，"
        "提到框架只能当例子（写成「例如…」）。\n"
        "② 第二层「知识库中的具体实现」：知识库里有哪些具体项目/文档、它们在这一点上怎么写、缺哪些。\n"
        "不要把某个具体项目或框架的写法当成通用定义，也不要用具体项目的内容代替通用解释。"
    ),
    "mixed": (
        "【知识边界】问题既涉及通用概念、又要落到具体项目。先给通用说明（标明「通用说明」），"
        "再用知识库里的具体实现印证；两层要分清，不要混着讲。"
    ),
}


def parse_plan(raw: str) -> Dict[str, Any]:
    """容错解析规划 JSON，任何异常都退回 direct/kb"""
    default = {"strategy": "direct", "sub_queries": [], "scope": "kb", "reason": ""}
    match = _JSON_OBJ_RE.search(raw or "")
    if not match:
        return default
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return default

    strategy = str(data.get("strategy") or "direct").strip().lower()
    if strategy not in ("direct", "hyde", "decompose"):
        strategy = "direct"
    scope = str(data.get("scope") or "kb").strip().lower()
    if scope not in ("kb", "general", "mixed"):
        scope = "kb"
    subs = [
        str(s).strip()
        for s in (data.get("sub_queries") or [])
        if str(s).strip()
    ][:4]
    return {
        "strategy": strategy,
        "sub_queries": subs,
        "scope": scope,
        "reason": str(data.get("reason") or "")[:120],
    }


def merge_hits(*lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """多路检索结果合并：同 (文档, 分块) 取高分，按分排序"""
    merged: Dict[tuple, Dict[str, Any]] = {}
    for hits in lists:
        for hit in hits:
            key = (int(hit["document_id"]), int(hit.get("chunk_index") or 0))
            prev = merged.get(key)
            if prev is None or float(hit.get("score") or 0) > float(prev.get("score") or 0):
                merged[key] = hit
    return sorted(merged.values(), key=lambda h: -float(h.get("score") or 0))


def check_answer(
    question: str, answer: str, chunks: List[Dict[str, Any]], fmt: str, scope: str = ""
) -> List[str]:
    """
    生成后的低成本自检（纯规则，0 成本）

    只查"机器判得准"的四类问题，判不准的（逻辑是否连贯）交给模型重写——
    所以这里不追求覆盖率，追求零误报：误报会白花一次重写。
    """
    issues: List[str] = []
    if not answer.strip():
        return ["empty"]
    if not chunks:
        return issues

    # 1. 引用编号越界：模型引用了不存在的资料序号（最常见的是凭记忆编号）
    cited = [int(n) for n in re.findall(r"\[(\d{1,2})\]", answer)]
    if cited and max(cited) > len(chunks):
        issues.append(
            f"引用编号越界：答案引用了 [{max(cited)}]，但资料只有 {len(chunks)} 条"
        )

    # 2. 重复句：同一句话换个标点又写一遍（中文一句常不到 10 字，阈值取 6）
    sents = [
        re.sub(r"\s+", "", s.strip())
        for s in _SENT_SPLIT.split(answer)
        if len(s.strip()) >= 6
    ]
    if sents:
        dup = len(sents) - len(set(sents))
        if dup >= 1 and dup / len(sents) >= 0.15:
            issues.append(f"重复表述：{len(sents)} 句里有 {dup} 句是重复的")

    # 3. 篇幅与证据不匹配：资料没那么多内容，答案却写了很长，多半是自行发挥
    context_chars = sum(len(c.get("content") or "") for c in chunks)
    if fmt == "keywords" and len(answer) > 150:
        issues.append(f"形式不符：要求关键词，实际写了 {len(answer)} 字")
    elif fmt == "short" and len(answer) > 120:
        issues.append(f"形式不符：要求简短，实际写了 {len(answer)} 字")
    elif context_chars and len(answer) > max(500, context_chars * 0.8):
        issues.append(
            f"超出资料体量：答案 {len(answer)} 字，参考资料合计只有 {context_chars} 字"
        )

    # 4. 跨文档拼接：不是对比类问题，却把 3 篇以上文档的内容混成一段
    #    通用概念题本来就该"概念 + 各项目实例"，引用多篇文档是正常的，不算毛病；
    #    短答案引 3 篇多半是"总览引用"（上下文本身最多 3 篇），只有长答案才谈得上拼接。
    docs = {c["document_id"] for c in chunks}
    q = question or ""
    comparison = any(w in q for w in ("对比", "比较", "区别", "哪些文档", "分别", "异同"))
    if (
        len(docs) >= 3
        and not comparison
        and scope not in ("general", "mixed")
        and len(answer) >= 800
    ):
        issues.append(f"跨文档拼接：资料来自 {len(docs)} 篇不同文档，且问题并未要求对比")

    return issues


def _top_vector_score(chunks: List[Dict[str, Any]]) -> float:
    """最高向量分（用原始向量分，不用加了文件名 boost 的最终分）"""
    return max(
        (float(c.get("vector_score") if c.get("vector_score") is not None else c.get("score") or 0)
         for c in chunks),
        default=0.0,
    )


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
        fmt: str = "default",
        low_relevance: bool = False,
        outline: str = "",
        scope: str = "",
    ) -> str:
        """组装 human 消息：对话历史 + 参考资料 + 提纲 + 输出形式 + 问题"""
        blocks: List[str] = []
        if history:
            lines = []
            for item in history:
                who = "用户" if item.get("role") == "user" else "助手"
                lines.append(f"{who}：{item.get('content', '')}")
            blocks.append("【对话历史】\n" + "\n".join(lines))
        blocks.append("【参考资料】\n" + context)
        scope_hint = _SCOPE_HINTS.get(scope or "")
        if scope_hint:
            blocks.append(scope_hint)
        if outline:
            blocks.append(_OUTLINE_HINT.format(outline=outline))
        hint = _FORMAT_HINTS.get(fmt) or ""
        if hint:
            blocks.append(hint)
        if low_relevance:
            blocks.append(
                "【注意】本次检索到的参考资料与问题的相关度较低。"
                "如果它们确实回答不了这个问题，就直接说明「知识库中没有相关内容」，"
                "不要勉强用不相关的材料拼凑答案。"
            )
        blocks.append("【问题】\n" + question)
        blocks.append("请按要求作答。")
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
        plan: Optional[Dict[str, Any]] = None,
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
        plan = plan or {}

        hint = ""
        if plan.get("strategy") == "decompose" and plan.get("sub_queries"):
            hint = (
                f"【规划建议】这个问题有多个子目标，先分别检索："
                f"{'、'.join(plan['sub_queries'][:4])}\n"
            )
        elif plan.get("strategy") == "hyde":
            hint = "【规划建议】问题偏抽象，检索时多换几种说法，不要只用原句。\n"
        if plan.get("scope") in ("general", "mixed"):
            hint += (
                "【规划建议】用户问的是通用概念，知识库里可能只有某个具体实例。"
                "检索时不要只认项目名，把概念本身也当检索词试一次。\n"
            )

        llm = self.llm.bind_tools(_LOOP_TOOLS)
        messages = [
            SystemMessage(content=_RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"{self._history_text(history)}"
                    f"{hint}"
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
                before = len(toolbox.evidence)
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

                # 这一轮检索没有带来任何新片段：模型在重复查同一件事，停手
                if step >= 2 and len(toolbox.evidence) == before:
                    steps.append(
                        {"step": step, "tool": "early_stop", "args": {},
                         "observation": "本轮无新增片段，停止检索"}
                    )
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
    # 这些问法天然需要多轮检索，不适用"证据够了就跳过循环"的早退
    _MULTI_GOAL_WORDS = (
        "对比", "比较", "分别", "区别", "差异", "哪些", "列出", "梳理",
        "总结", "优缺点", "几个方面",
    )
    _REFER_WORDS = ("那", "这", "它", "他们", "其中", "上面", "刚才", "前面")

    def _evidence_enough(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        plan: Optional[Dict[str, Any]],
    ) -> bool:
        """
        证据已经足够好时，不必再进工具循环

        qa_agent_always 打开后所有问题都会走循环，代价是每次多 2~3 轮模型往返
        （5~10 秒）；但"问题就是某份文档里的某件事、首轮检索已经命中"这类情况，
        循环只是把同样的片段再捡一遍。这里用可解释的条件提前收手：
        规划是 direct、命中分数够高、片段数够多、问法不是"对比/列举"。
        """
        if not settings.qa_agent_early_exit_enabled:
            return False
        if not chunks:
            return False
        if (plan or {}).get("strategy") != "direct":
            return False
        q = question or ""
        if any(w in q for w in self._MULTI_GOAL_WORDS):
            return False
        if len(chunks) < int(settings.qa_agent_early_exit_chunks):
            return False
        try:
            top = max(float(c.get("score") or 0.0) for c in chunks)
        except (TypeError, ValueError):
            return False
        return top >= float(settings.qa_agent_early_exit_score)

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
        # 只要关键词/一句话的请求，单次检索就够了：进循环等于为 5 个词把整篇文档读一遍
        # （实测"总结《…》只要 5 个关键词"走了 5 步、读了 22 个片段，5.9 秒）
        if detect_format(q) in ("keywords", "short"):
            return False

        if any(word in q for word in self._TRIGGER_WORDS):
            return True

        if history and (len(q) <= 12 or any(q.startswith(w) for w in self._REFER_WORDS)):
            return True

        # 重排会改变顺序，所以取"全体的最高分"而不是第一条的分数
        try:
            top = max(float(c.get("score") or 0.0) for c in chunks)
        except (TypeError, ValueError):
            top = 0.0
        return top < float(settings.qa_agent_score_gate)

    # ---------- 检索规划 ----------
    def _plan_query(
        self, question: str, history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """让模型决定：直接查 / HyDE / 拆子问题，以及问题属于通用概念还是具体项目"""
        if not settings.qa_query_plan_enabled:
            return {"strategy": "direct", "sub_queries": [], "scope": "kb", "reason": "规划已关闭"}

        hist = ""
        if history:
            lines = [f"{'用户' if m.get('role') == 'user' else '助手'}：{m.get('content', '')[:80]}" for m in history[-4:]]
            hist = "【对话历史】\n" + "\n".join(lines) + "\n\n"

        try:
            raw = self.invoke_llm(
                _QUERY_PLAN_TEMPLATE.format(history=hist, question=question),
                system=_QUERY_PLAN_SYSTEM,
                max_tokens=200,
            )
            plan = parse_plan(raw)
        except Exception as exc:  # 规划失败就退回直接检索，不能挡住问答
            logger.warning(f"检索规划失败，退回直接检索: {exc}")
            return {"strategy": "direct", "sub_queries": [], "scope": "kb", "reason": "规划失败"}

        # 规则兜底：问的是概念，就不能被规划成"某个具体项目"的题。
        # 模型判层级不稳定（同一个问题两次分别给出 general/mixed），所以补一条可解释的规则：
        # 有概念性问法、又没指向具体文档/项目时，一律按 general 处理。
        q = question or ""
        conceptual = any(
            w in q
            for w in (
                "什么是", "是什么", "讲讲", "讲解", "介绍一下", "解释",
                "详细说", "怎么理解", "一般怎么", "通常怎么", "如何选", "怎么选",
            )
        )
        doc_ref = "《" in q or any(
            w in q for w in ("简历", "试卷", "面试", "EduAgent", "ResumeAgent", "这篇", "文档中")
        )
        if conceptual and not doc_ref and plan["scope"] != "general":
            plan["scope"] = "general"
            plan["reason"] = (plan["reason"] + "｜规则：概念性提问")[:120]

        # 规则兜底：操作性提问（怎么用/步骤/教程）措辞和资料几乎一致，
        # 拆子问题或假设答案只会把检索词改得离原文更远，直接查更快更准。
        howto = any(
            w in q
            for w in (
                "怎么用", "如何使用", "怎么使用", "如何操作", "怎么操作",
                "操作步骤", "使用步骤", "怎么弄", "教程",
            )
        )
        if howto and plan["strategy"] != "direct":
            plan["strategy"] = "direct"
            plan["sub_queries"] = []
            plan["reason"] = (plan["reason"] + "｜规则：操作性提问")[:120]

        logger.info(
            f"检索规划：策略={plan['strategy']} 层级={plan['scope']} "
            f"子问题={len(plan['sub_queries'])} 理由={plan['reason'][:40]}"
        )
        return plan

    def _hypothetical_answer(self, question: str) -> str:
        """HyDE：先让模型写一段"像样的答案"，用它去检索，弥补问题与资料措辞的差距"""
        try:
            return self.invoke_llm(
                _HYDE_TEMPLATE.format(question=question),
                system=_HYDE_SYSTEM,
                max_tokens=int(settings.qa_hyde_max_tokens),
            ).strip()
        except Exception as exc:
            logger.warning(f"HyDE 生成失败，退回原问题检索: {exc}")
            return ""

    def _retrieve_planned(
        self,
        question: str,
        plan: Dict[str, Any],
        top_k: int,
        document_ids: Optional[List[int]],
        use_name_match: bool,
    ) -> List[Dict[str, Any]]:
        """按规划执行检索，多路结果合并去重"""
        from concurrent.futures import ThreadPoolExecutor

        strategy = plan.get("strategy") or "direct"

        if strategy == "hyde":
            # 假设答案的生成和首轮检索互不依赖：并行跑，省掉一次串行等待
            with ThreadPoolExecutor(max_workers=2) as pool:
                base_future = pool.submit(
                    self._retrieve, question, top_k, document_ids, use_name_match
                )
                hypo = self._hypothetical_answer(question)
                base_hits = base_future.result()
            if hypo:
                logger.info(f"HyDE 检索词：{hypo[:60]}")
                return merge_hits(
                    base_hits,
                    self._retrieve(hypo, top_k, document_ids, use_name_match),
                )[: top_k * 2]
            return base_hits

        elif strategy == "decompose":
            subs = plan.get("sub_queries") or []
            if len(subs) >= 2:
                logger.info(f"拆分检索：{subs}")
                per = max(3, int(top_k))
                with ThreadPoolExecutor(max_workers=min(4, len(subs))) as pool:
                    lists = list(
                        pool.map(
                            lambda s: self._retrieve(
                                s, per, document_ids, use_name_match
                            ),
                            subs,
                        )
                    )
                return merge_hits(*lists)[: top_k * 2]

        return self._retrieve(question, top_k, document_ids, use_name_match)

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

        # 分诊先于检索：寒暄/身份类问题没有资料可查，不该为它付检索、
        # 规划、循环、提纲的钱（实测一次寒暄要走 5 次模型往返、20 秒）
        intent = detect_intent(question)
        if intent:
            return {
                "chunks": [],
                "steps": [
                    {
                        "step": "intent",
                        "tool": "router",
                        "args": {"intent": intent},
                        "observation": "寒暄/身份类，直接答，不检索",
                    }
                ],
                "queries": [],
                "gave_up": False,
                "degraded": False,
                "social": intent,
                "plan": {
                    "strategy": "direct",
                    "sub_queries": [],
                    "scope": "kb",
                    "reason": "寒暄/身份类",
                },
            }

        # 形式类请求（关键词/一句话）本来就走单次检索，没必要先花一次规划调用
        fmt = detect_format(question)
        if fmt in ("keywords", "short") and not force_agent:
            plan: Dict[str, Any] = {
                "strategy": "direct",
                "sub_queries": [],
                "scope": "kb",
                "reason": "形式类请求，直接检索",
            }
        else:
            plan = self._plan_query(question, history)
        base = self._retrieve_planned(question, plan, top_k, document_ids, use_name_match)

        if not settings.qa_agent_enabled and not force_agent:
            solved: Dict[str, Any] = {
                "chunks": base,
                "steps": [],
                "queries": [],
                "gave_up": False,
                "degraded": True,
            }
        elif not force_agent and fmt in ("keywords", "short"):
            solved = {
                "chunks": base,
                "steps": [],
                "queries": [],
                "gave_up": False,
                "gated": True,
            }
        elif not force_agent and self._evidence_enough(question, base, plan):
            # 证据已经足够：再走循环只是多花 5~10 秒把同样的片段再捡一遍
            solved = {
                "chunks": base,
                "steps": [],
                "queries": [],
                "gave_up": False,
                "gated": True,
            }
        elif (
            not force_agent
            and not settings.qa_agent_always
            and not self._needs_agent(question, base, history)
        ):
            solved = {
                "chunks": base,
                "steps": [],
                "queries": [],
                "gave_up": False,
                "gated": True,
            }
        else:
            try:
                solved = self.solve(
                    question,
                    history=history,
                    top_k=top_k,
                    document_ids=document_ids,
                    use_name_match=use_name_match,
                    plan=plan,
                )
            except Exception as exc:
                logger.warning(f"agent 循环失败，降级为单次检索: {exc}")
                solved = {
                    "chunks": base,
                    "steps": [],
                    "queries": [],
                    "gave_up": False,
                    "degraded": True,
                }

        solved["plan"] = plan
        solved["steps"] = list(solved.get("steps") or [])
        solved["steps"].insert(
            0,
            {
                "step": "plan",
                "tool": "query_plan",
                "args": {
                    "strategy": plan.get("strategy"),
                    "scope": plan.get("scope"),
                    "sub_queries": plan.get("sub_queries"),
                },
                "observation": plan.get("reason") or "",
            },
        )

        # 统一在这里重排：所有调用方（同步/流式/引用）拿到的顺序必须一致，
        # 否则提示词里的编号和 citations 里的编号会对不上
        solved["chunks"] = order_chunks(solved["chunks"])
        solved["chunks"] = self._expand_neighbors(solved["chunks"])
        return solved

    # ---------- 相邻片段补全 ----------
    def _expand_neighbors(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        给排名靠前的文档补上命中片段的前后邻居

        检索只返回"最像的块"，段落经常从半句开始、到半句结束，模型读起来是断的。
        这里按 (document_id, chunk_index±1) 去库里把邻居捞出来，再统一重排，
        让每个文档的上下文变成一段连续的原文。每个文档最多补 _CTX_NEIGHBORS_PER_DOC 块。
        """
        if not chunks:
            return chunks

        from app.db.database import session_scope
        from app.models.document import Chunk

        groups: Dict[int, List[Dict[str, Any]]] = {}
        for c in chunks:
            groups.setdefault(int(c["document_id"]), []).append(c)
        ranked = sorted(
            groups.items(),
            key=lambda kv: -max(float(c.get("score") or 0) for c in kv[1]),
        )

        wanted: Dict[int, List[int]] = {}
        for doc_id, items in ranked[:_CTX_MAX_DOCS]:
            have = {int(c.get("chunk_index") or 0) for c in items}
            need: List[int] = []
            for idx in sorted(have):
                for nb in (idx - 1, idx + 1):
                    if nb >= 0 and nb not in have and nb not in need:
                        need.append(nb)
            if need:
                wanted[doc_id] = need[:_CTX_NEIGHBORS_PER_DOC]

        if not wanted:
            return chunks

        try:
            with session_scope() as db:
                rows = (
                    db.query(Chunk)
                    .filter(
                        Chunk.document_id.in_(list(wanted.keys()))
                    )
                    .all()
                )
            by_key = {(r.document_id, r.chunk_index): r for r in rows}
        except Exception as exc:  # 补邻居失败不能影响回答
            logger.warning(f"补相邻片段失败: {exc}")
            return chunks

        # 邻居沿用同一文档已有片段的标题/文件名，分数记 0（它不是因为相似进来的）
        meta = {doc: items[0] for doc, items in groups.items()}
        extra: List[Dict[str, Any]] = []
        for doc_id, idxs in wanted.items():
            base = meta[doc_id]
            for idx in idxs:
                row = by_key.get((doc_id, idx))
                if row is None:
                    continue
                extra.append(
                    {
                        "document_id": doc_id,
                        "chunk_index": idx,
                        "content": row.content,
                        "title": base.get("title"),
                        "filename": base.get("filename"),
                        "score": 0.0,
                        "vector_score": 0.0,
                        "name_score": 0.0,
                        "match_type": "neighbor",
                    }
                )

        if not extra:
            return chunks
        logger.debug(f"补入 {len(extra)} 个相邻片段")
        return order_chunks(chunks + extra)

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
        steps = list(solved["steps"])

        # 寒暄/身份类：固定话术直接返回，不检索、不生成、不带引用
        if solved.get("social"):
            return {
                "question": question,
                "answer": social_reply(solved["social"]),
                "citations": [],
                "retrieved": 0,
                "steps": steps,
                "queries": [],
            }

        if not chunks:
            return {
                "question": question,
                "answer": "知识库中暂未检索到与该问题相关的内容。请先上传相关文档，或换一种问法试试。",
                "citations": [],
                "retrieved": 0,
                "steps": steps,
                "queries": solved["queries"],
            }

        # 硬闸门：最强的一条都低于下限，说明知识库里确实没有相关内容。
        # 直接回固定话术，省掉一次 2~8 秒的生成，也避免模型硬凑。
        top_score = _top_vector_score(chunks)
        gave_up = bool(solved.get("gave_up"))
        if top_score < float(settings.qa_answer_floor) or (
            # 模型自己放弃，且检索分也不高：它已经判断"搜不到"，就别再让它硬写
            gave_up
            and top_score < float(settings.qa_agent_score_gate)
        ):
            logger.info(
                f"拒绝作答（top1={top_score:.3f}, gave_up={gave_up}）：'{question[:24]}'"
            )
            return {
                "question": question,
                "answer": (
                    "知识库中没有找到与该问题相关的内容。"
                    "可以换一种问法，或先上传相关文档。"
                ),
                "citations": [],
                "retrieved": len(chunks),
                "steps": steps
                + [
                    {
                        "step": "gate",
                        "tool": "low_relevance",
                        "args": {
                            "top1": round(top_score, 3),
                            "gave_up": gave_up,
                        },
                        "observation": "相关度低于下限或模型已放弃检索，未作答",
                    }
                ],
                "queries": solved["queries"],
            }

        low_relevance = top_score < float(settings.qa_low_relevance_gate)
        fmt = detect_format(question)
        scope = (solved.get("plan") or {}).get("scope", "")
        context = self._build_context(chunks)
        outline = self._plan_outline(question, context, fmt)
        answer = self._run_chain(
            question,
            context,
            history,
            low_relevance=low_relevance,
            fmt=fmt,
            outline=outline,
            scope=scope,
        )
        if outline:
            steps.append(
                {
                    "step": "outline",
                    "tool": "plan",
                    "args": {"lines": len([x for x in outline.splitlines() if x.strip()])},
                    "observation": outline[:120],
                }
            )

        # 生成后自检：规则命中才重写一次（自检本身不花钱，重写才花）
        answer, review = self._review_and_repair(question, answer, chunks, scope)
        if review["triggered"] or review["issues"]:
            steps.append(
                {
                    "step": "review",
                    "tool": "self_check",
                    "args": {"issues": review["issues"]},
                    "observation": "已重写" if review["repaired"] else "未重写",
                }
            )

        return {
            "question": question,
            "answer": answer,
            "citations": self._citations(chunks),
            "retrieved": len(chunks),
            "steps": steps,
            "queries": solved["queries"],
            "gave_up": solved.get("gave_up", False),
            "degraded": solved.get("degraded", False),
        }

    def _review_and_repair(
        self, question: str, answer: str, chunks: List[Dict[str, Any]], scope: str = ""
    ) -> tuple:
        """
        生成后自检 + 条件重写

        为什么不直接"让模型自己检查一遍"：
            - 无 rubric 的自评几乎总是"我觉得没问题"，白花一次调用；
            - 所以先用规则查机器判得准的毛病（引用越界/重复句/超出资料体量/跨文档拼接），
              只有命中才叫模型，并且把"具体哪里有问题"作为输入交回去；
            - 重写只做 1 次，避免"改坏→再改"的抖动。
        """
        fmt = detect_format(question)
        issues = check_answer(question, answer, chunks, fmt, scope)
        result = {"triggered": bool(issues), "issues": issues, "repaired": False}
        if not issues:
            return answer, result
        if not settings.qa_review_enabled:
            logger.info(f"答案自检命中但重写已关闭：{issues}")
            return answer, result

        # 重写阶段只喂最相关的几条片段，避免又一次把整库拖进来
        context = self._build_context(chunks[:6])
        try:
            fixed = self.invoke_llm(
                _REPAIR_TEMPLATE.format(
                    question=question,
                    answer=answer,
                    issues="\n".join(f"- {i}" for i in issues),
                    context=context,
                ),
                system=_REPAIR_SYSTEM,
                max_tokens=_FORMAT_MAX_TOKENS[fmt],
            ).strip()
        except Exception as exc:
            logger.warning(f"答案重写失败，沿用原文: {exc}")
            return answer, result

        # 重写反而更糟（变短一半以上/空）就不要了
        if not fixed or len(fixed) < len(answer) * 0.5:
            logger.info("重写结果异常，沿用原答案")
            return answer, result

        logger.info(
            f"答案已重写：{len(answer)}字 -> {len(fixed)}字，问题={issues}"
        )
        result["repaired"] = True
        return fixed, result

    def _plan_outline(self, question: str, context: str, fmt: str) -> str:
        """
        作答前先列提纲

        资料是按顺序拼的，模型逐段看、逐段写，最后常常是"每段都合理、
        放一起没逻辑"。先让它定一个能自洽的骨架（结论 → 支撑 → 边界），
        再按骨架填内容，比事后重写有效得多。
        """
        if not settings.qa_outline_enabled:
            return ""
        if fmt not in ("default", "detailed"):
            return ""
        if len(context) < 600:  # 资料太少，没什么可规划的
            return ""
        try:
            outline = self.invoke_llm(
                _OUTLINE_TEMPLATE.format(context=context, question=question),
                system=_OUTLINE_SYSTEM,
                max_tokens=int(settings.qa_outline_max_tokens),
            )
        except Exception as exc:
            logger.warning(f"生成提纲失败，直接作答: {exc}")
            return ""
        return outline.strip()

    def _run_chain(
        self,
        question: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None,
        low_relevance: bool = False,
        fmt: Optional[str] = None,
        outline: str = "",
        scope: str = "",
    ) -> str:
        """LangChain LCEL 链：Prompt | LLM(~max_tokens) | StrOutputParser"""
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        fmt = fmt or detect_format(question)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT),
                ("human", _USER_TEMPLATE),
            ]
        )
        # 按形式卡输出预算：不卡的话全局 2048 会让"给我几个关键词"写成一千多字
        llm = self.llm.bind(max_tokens=_FORMAT_MAX_TOKENS[fmt])
        chain = prompt | llm | StrOutputParser()
        return chain.invoke(
            {
                "prompt": self._build_prompt(
                    context, question, history, fmt, low_relevance, outline, scope
                )
            }
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

        # 寒暄/身份类：固定话术，不走检索与生成
        social = kwargs.get("social") or ""
        if social:
            yield social_reply(social)
            return

        top_k = int(kwargs.get("top_k") or settings.search_top_k)
        document_ids = kwargs.get("document_ids")
        use_name_match = kwargs.get("use_name_match", True)
        history = kwargs.get("history") or []
        low_relevance = bool(kwargs.get("low_relevance", False))
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
        fmt = detect_format(question)
        # 提纲在 prepare_stream 里已经算过（那时更早，能早点把首字推出去）；
        # 单独调 astream 时才现算，保证两条路径行为一致
        outline = kwargs.get("outline") or self._plan_outline(question, context, fmt)
        prompt = ChatPromptTemplate.from_messages(
            [("system", _SYSTEM_PROMPT), ("human", _USER_TEMPLATE)]
        )
        llm = self.llm.bind(max_tokens=_FORMAT_MAX_TOKENS[fmt])
        chain = prompt | llm | StrOutputParser()

        payload = {
            "prompt": self._build_prompt(
                context,
                question,
                history,
                fmt,
                low_relevance,
                outline,
                str(kwargs.get("scope") or ""),
            )
        }
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
        plan = solved.get("plan") or {}
        top_score = _top_vector_score(chunks)
        gave_up = bool(solved.get("gave_up"))
        fmt = detect_format(question)
        outline = self._plan_outline(question, self._build_context(chunks), fmt) if chunks else ""
        return {
            "chunks": chunks,
            "citations": self._citations(chunks) if chunks else [],
            "steps": solved["steps"],
            "queries": solved["queries"],
            "degraded": solved.get("degraded", False),
            "top_score": round(top_score, 4),
            "low_relevance": top_score < float(settings.qa_low_relevance_gate),
            "outline": outline,
            "scope": plan.get("scope", ""),
            "plan": plan,
            "social": solved.get("social") or "",
            "answer": social_reply(solved["social"]) if solved.get("social") else "",
            "refusal": bool(chunks)
            and (
                top_score < float(settings.qa_answer_floor)
                or (gave_up and top_score < float(settings.qa_agent_score_gate))
            ),
        }

    def review_answer(
        self, question: str, answer: str, chunks: List[Dict[str, Any]], scope: str = ""
    ) -> tuple:
        """流式路径用：前端已经看到初稿，这里返回 (修正稿, 自检说明)"""
        return self._review_and_repair(question, answer, chunks, scope)

    def health(self) -> Dict[str, Any]:
        ok = milvus_store.health()
        return {"agent": self.name, "milvus": ok, "status": "ok" if ok else "down"}


qa_agent = QAAgent()
