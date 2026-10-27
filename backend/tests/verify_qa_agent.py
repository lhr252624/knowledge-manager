"""
QA Agent 单元测试

分两层：
    A. 闸门逻辑（确定性，不调 LLM / Milvus）：直接测 _needs_agent
    B. 集成（调 LLM / Milvus）：验证 collect / run / prepare_stream 的降级与预算

跑法：
    cd backend
    .\\venv\\Scripts\\python.exe tests\\verify_qa_agent.py
"""
import sys
from pathlib import Path

backend = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend))

from app.agents.qa_agent import qa_agent  # noqa: E402
from app.config import get_settings  # noqa: E402

settings = get_settings()


def _chunks(score: float = 0.9):
    """造一条假检索结果，供 _needs_agent 用"""
    return [{"document_id": 1, "chunk_index": 0, "content": "x", "score": score}]


# ================= A. 闸门（确定性） =================

def test_gate_trigger_words():
    """触发词：单次检索结构上做不到的需求，必须进循环"""
    for q in ("总结一下所有文档", "概述全文", "通读这篇", "对比两个方案",
              "区别在哪里", "哪个更好", "都有哪些文档", "有哪些结论",
              "列出所有", "分别有哪些"):
        assert qa_agent._needs_agent(q, _chunks(0.9), None) is True, f"触发词未命中：{q}"
    # 「比较」刻意不在触发词里："比较的损失函数是哪一个"是问值，不是要对比
    assert qa_agent._needs_agent("比较的损失函数是哪一个", _chunks(0.9), None) is False
    print("OK 触发词：含总结/对比/列出… 进循环，「比较」不进")


def test_gate_short_followup():
    """短追问 + 历史：需要先把指代展开，进循环"""
    history = [{"role": "user", "content": "损失函数有哪些"},
               {"role": "assistant", "content": "交叉熵、MSE、MAE"}]
    for q in ("那第二个呢", "它呢", "这个怎么算", "其中第一点"):
        assert qa_agent._needs_agent(q, _chunks(0.9), history) is True, f"短追问未命中：{q}"
    # 没有历史时，同样的短句不该触发（无上下文可展开）
    assert qa_agent._needs_agent("那第二个呢", _chunks(0.9), None) is False
    print("OK 短追问：有历史时进循环，无历史不进")


def test_gate_score():
    """分数闸门：高分直出，低分升级"""
    assert qa_agent._needs_agent("月结余多少", _chunks(0.9), None) is False  # 高分直出
    assert qa_agent._needs_agent("月结余多少", _chunks(0.3), None) is True   # 低分升级
    assert qa_agent._needs_agent("月结余多少", _chunks(0.55), None) is False  # 恰好等于阈值，直出
    print("OK 分数闸门：高分/等于阈值直出，低分升级")


def test_gate_empty():
    """检索零命中：必须进循环（换措辞重搜）"""
    assert qa_agent._needs_agent("任何问题", [], None) is True
    print("OK 零命中：进循环")


# ================= B. 集成（调 LLM / Milvus） =================

def test_agent_disabled_degrade():
    """QA_AGENT_ENABLED=False：即使触发词也退回单次检索，标记 degraded"""
    old = settings.qa_agent_enabled
    try:
        settings.qa_agent_enabled = False
        result = qa_agent.collect(question="总结所有文档", history=None, top_k=5)
        assert result["steps"] == [], "agent 关闭时 steps 应为空"
        assert result.get("degraded") is True, "应标记为降级"
        assert len(result["chunks"]) > 0, "降级后仍应有检索结果"
        print(f"OK agent 关闭降级：{len(result['chunks'])} 片段，degraded={result['degraded']}")
    finally:
        settings.qa_agent_enabled = old


def test_force_agent_and_budget():
    """强制 agent + 轮次预算：steps 非空，且轮次数不超过 max_steps"""
    old = settings.qa_agent_max_steps
    try:
        settings.qa_agent_max_steps = 2
        result = qa_agent.collect(
            question="列出所有文档并分别总结",
            history=None, top_k=5, force_agent=True,
        )
        steps = result.get("steps", [])
        rounds = {s.get("step", 1) for s in steps}
        assert len(steps) > 0, "强制 agent 应产生步骤"
        assert max(rounds, default=0) <= 2, f"超出轮次预算：{rounds}"
        print(f"OK 强制 agent + 预算：{len(steps)} 个工具调用 / {max(rounds, default=0)} 轮（上限 2）")
    finally:
        settings.qa_agent_max_steps = old


def test_run_sync():
    """同步 run() 返回 answer / citations / steps / queries 字段"""
    result = qa_agent.run(question="深度学习的优化器有哪些", top_k=5)
    assert isinstance(result.get("answer"), str) and result["answer"], "必须有 answer"
    assert isinstance(result.get("citations"), list), "citations 应为 list"
    assert "steps" in result and "queries" in result, "应返回 steps 与 queries"
    print(f"OK 同步 run：{len(result['answer'])} 字，{len(result['citations'])} 引用，"
          f"{len(result['steps'])} 步")


def test_empty_question():
    """边界：空问题抛 ValueError"""
    try:
        qa_agent.run(question="")
        raise AssertionError("空问题应抛异常")
    except ValueError as e:
        assert "不能为空" in str(e)
        print("OK 空问题拒绝")


def test_prepare_stream():
    """prepare_stream 返回 chunks / citations / steps"""
    meta = qa_agent.prepare_stream(question="深度学习的基本概念", history=None, top_k=5)
    for key in ("chunks", "citations", "steps"):
        assert key in meta, f"缺字段 {key}"
    print(f"OK prepare_stream：{len(meta['chunks'])} 片段，{len(meta['steps'])} 步")


# ================= 执行 =================

if __name__ == "__main__":
    tests = [
        ("闸门-触发词", test_gate_trigger_words),
        ("闸门-短追问", test_gate_short_followup),
        ("闸门-分数", test_gate_score),
        ("闸门-零命中", test_gate_empty),
        ("降级-agent关闭", test_agent_disabled_degrade),
        ("强制agent+预算", test_force_agent_and_budget),
        ("同步run", test_run_sync),
        ("边界-空问题", test_empty_question),
        ("prepare_stream", test_prepare_stream),
    ]

    print("=" * 60)
    print("  QA Agent 单元测试")
    print("=" * 60)
    passed = 0
    failed = []
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"X {name}: {e}")
            failed.append(name)
        except Exception as e:
            print(f"X {name}: 异常 {type(e).__name__}: {e}")
            failed.append(name)

    print("=" * 60)
    print(f"  通过 {passed}/{len(tests)}")
    if failed:
        print(f"  失败：{', '.join(failed)}")
        sys.exit(1)
    print("  全部通过")
    print("=" * 60)
