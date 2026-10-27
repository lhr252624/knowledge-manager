"""检索与问答 API（含语义缓存与历史对话落库）"""
import json
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from app.agents.qa_agent import qa_agent
from app.agents.search_agent import search_agent
from app.api.schemas import QARequest, QAResponse, SearchRequest
from app.services import conversation_service
from app.services.qa_cache_service import qa_cache_service

router = APIRouter(tags=["检索与问答"])


@router.post("/search", summary="语义检索")
def search(req: SearchRequest):
    try:
        return search_agent.run(
            query=req.query,
            top_k=req.top_k,
            document_ids=req.document_ids,
            group_by_document=req.group_by_document,
            use_name_match=req.use_name_match,
        )
    except Exception as exc:
        logger.exception("检索失败")
        raise HTTPException(status_code=500, detail=f"检索失败: {exc}") from exc


# ---------- 问答公共逻辑 ----------
def _prepare_conversation(req: QARequest) -> Optional[int]:
    """定位或新建会话，返回会话 id；不保存历史时返回 None"""
    if not req.save_history:
        return None

    if req.conversation_id:
        existing = conversation_service.get_conversation(
            req.conversation_id, with_messages=False
        )
        if existing is not None:
            return int(existing["id"])

    created = conversation_service.create_conversation(req.question)
    return int(created["id"])


def _try_cache(req: QARequest) -> Dict[str, Any]:
    """
    查缓存

    限定 document_ids 的查询不动用全局缓存：同一个问题在不同文档范围下
    答案可能不同，命中全局缓存会答非所问。
    """
    empty: Dict[str, Any] = {"hit": None, "near": None, "method": None, "elapsed_ms": 0.0}
    if not req.use_cache or req.document_ids:
        return empty
    try:
        return qa_cache_service.lookup_ex(req.question)
    except Exception as exc:  # 缓存故障不能影响问答
        logger.warning(f"查询问答缓存失败，直接走模型: {exc}")
        return empty


def _cache_payload(hit: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "hit": True,
        "method": hit.get("method"),
        "similarity": hit["similarity"],
        "confidence": hit.get("confidence"),
        "score": hit.get("score"),
        "matched_question": hit.get("matched_question"),
        "exact": bool(hit.get("exact")),
    }


def _persist(
    conversation_id: Optional[int],
    req: QARequest,
    answer: str,
    citations: list,
    hit: Optional[Dict[str, Any]],
    latency_ms: int,
) -> None:
    if conversation_id:
        try:
            conversation_service.add_user_and_assistant(
                conversation_id=conversation_id,
                question=req.question,
                answer=answer,
                citations=citations,
                cached=bool(hit),
                similarity=hit["similarity"] if hit else None,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            logger.warning(f"保存对话消息失败: {exc}")


@router.post("/qa", response_model=QAResponse, summary="知识库问答")
def question_answer(req: QARequest, background_tasks: BackgroundTasks):
    started = time.perf_counter()
    conversation_id = _prepare_conversation(req)
    history = (
        conversation_service.history_for_prompt(conversation_id, exclude_last=0)
        if conversation_id
        else []
    )

    cache_result = _try_cache(req)
    hit = cache_result["hit"]
    near = cache_result.get("near")

    if hit is not None:
        answer = hit["answer"]
        citations = hit["citations"] or []
        background_tasks.add_task(
            qa_cache_service.mark_hit, hit["cache_id"], hit["similarity"]
        )
        logger.info(
            f"问答命中缓存：'{req.question[:24]}' 方式 {hit.get('method')} "
            f"相似度 {hit['similarity']}"
        )
    else:
        try:
            result = qa_agent.run(
                question=req.question,
                top_k=req.top_k,
                document_ids=req.document_ids,
                use_name_match=req.use_name_match,
                history=history,
            )
        except Exception as exc:
            logger.exception("问答失败")
            raise HTTPException(status_code=500, detail=f"问答失败: {exc}") from exc

        answer = result["answer"]
        citations = result.get("citations") or []
        if req.use_cache and not req.document_ids:
            try:
                qa_cache_service.store(req.question, answer, citations)
            except Exception as exc:
                logger.warning(f"写入问答缓存失败: {exc}")

    latency_ms = int((time.perf_counter() - started) * 1000)
    _persist(conversation_id, req, answer, citations, hit, latency_ms)

    return {
        "question": req.question,
        "answer": answer,
        "citations": citations,
        "retrieved": len(citations),
        "conversation_id": conversation_id,
        "cached": hit is not None,
        "cache": _cache_payload(hit) if hit else None,
        "near": near,
        "steps": [] if hit is not None else (result.get("steps") or []),
        "queries": [] if hit is not None else (result.get("queries") or []),
        "degraded": False if hit is not None else bool(result.get("degraded")),
        "latency_ms": latency_ms,
    }


@router.post("/qa/stream", summary="知识库问答（流式）")
async def question_answer_stream(req: QARequest, background_tasks: BackgroundTasks):
    """
    SSE 流式返回：
        先发一条 meta 事件（引用来源 + 会话 id + 是否缓存命中），
        再逐段发 delta 事件，最后 done
    """
    started = time.perf_counter()
    conversation_id = _prepare_conversation(req)
    history = (
        conversation_service.history_for_prompt(conversation_id, exclude_last=0)
        if conversation_id
        else []
    )

    cache_result = _try_cache(req)
    hit = cache_result["hit"]
    near = cache_result.get("near")
    prepared: Dict[str, Any] = {"chunks": [], "citations": []}

    if hit is None:
        try:
            prepared = qa_agent.prepare_stream(
                question=req.question,
                top_k=req.top_k,
                document_ids=req.document_ids,
                use_name_match=req.use_name_match,
                history=history,
            )
        except Exception as exc:
            logger.exception("问答检索失败")
            raise HTTPException(status_code=500, detail=f"检索失败: {exc}") from exc

    if hit is not None:
        logger.info(
            f"问答命中缓存：'{req.question[:24]}' 方式 {hit.get('method')} "
            f"相似度 {hit['similarity']}"
        )

    meta = {
        "citations": hit["citations"] if hit is not None else prepared["citations"],
        "retrieved": len(hit["citations"] or []) if hit is not None else len(prepared["chunks"]),
        "conversation_id": conversation_id,
        "cached": hit is not None,
        "cache": _cache_payload(hit) if hit else None,
        "near": near,
    }
    if hit is None:
        meta["steps"] = prepared.get("steps") or []
        meta["degraded"] = bool(prepared.get("degraded"))

    async def event_generator():
        answer_parts = []
        yield _sse("meta", json.dumps(meta, ensure_ascii=False))

        # agent 的收集过程：前端可以据此显示"检索了什么、读了几段"
        # （收集在 meta 之前已经完成，这里只是逐条回放，不是实时推进）
        for step in (meta.get("steps") or []):
            yield _sse("step", json.dumps(step, ensure_ascii=False))

        if hit is not None:
            # 缓存命中也走同一套 delta 事件，前端不用区分两种渲染路径
            answer_parts.append(hit["answer"])
            yield _sse(
                "delta", json.dumps({"text": hit["answer"]}, ensure_ascii=False)
            )
            background_tasks.add_task(
                qa_cache_service.mark_hit, hit["cache_id"], hit["similarity"]
            )
        else:
            try:
                async for piece in qa_agent.astream(
                    question=req.question,
                    top_k=req.top_k,
                    document_ids=req.document_ids,
                    use_name_match=req.use_name_match,
                    history=history,
                    chunks=prepared["chunks"],
                ):
                    answer_parts.append(piece)
                    yield _sse("delta", json.dumps({"text": piece}, ensure_ascii=False))
            except Exception as exc:  # 流中断时通知前端
                logger.exception("流式回答中断")
                yield _sse("error", json.dumps({"message": str(exc)}, ensure_ascii=False))

        answer = "".join(answer_parts).strip()
        citations = hit["citations"] if hit is not None else prepared["citations"]

        if answer and hit is None and req.use_cache and not req.document_ids:
            try:
                qa_cache_service.store(req.question, answer, citations)
            except Exception as exc:
                logger.warning(f"写入问答缓存失败: {exc}")

        latency_ms = int((time.perf_counter() - started) * 1000)
        _persist(conversation_id, req, answer, citations, hit, latency_ms)
        yield _sse(
            "done",
            json.dumps(
                {"latency_ms": latency_ms, "conversation_id": conversation_id},
                ensure_ascii=False,
            ),
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"
