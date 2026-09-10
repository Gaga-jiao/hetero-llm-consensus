# -*- coding: utf-8 -*-
"""多源裁决组件 —— 可直接接进 Agent 的记忆写入 / 结论确认路径。

立场完全由实测数据决定（见 README）：
  * prompt 自报置信度 ECE = 0.537，且与正确率**负相关**（最自信的模型最不准）
    → 不用它做加权；置信度加权实测把准确率从 23% 打到 13%
  * 不同厂商模型之间错误相关性 phi = +0.58，同一模型多次采样 phi = +0.74
    → 「多来源」不等于「多视角」；同源读取时投票无效
  * 多数投票相对最好单模型**零增益**；自一致性是唯一单调有效的信号
    → 用它判断「结论可不可信」，而不是用它选答案
  * 低可信时的正确动作是**升级验证方式**（检索 / 跑工具 / 问人），
    而不是再多开几个模型

用法：
    from verdict import Source, verdict
    r = verdict([
        Source("deepseek", ["Michio Sugeno", "Michio Sugeno", "Sugeno"]),
        Source("zhipu", ["Michio Sugeno", "Sugeno", "unknown"]),
        Source("manual-note", ["Sugeno"], independent=False),
    ])
    print(r["answer"], r["risk"], r["warnings"])
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Source:
    name: str
    samples: list[str] = field(default_factory=list)
    independent: bool = True          # False = 与其它来源共享输入（同一份材料/同一检索结果）
    confidence: float | None = None   # 自报置信度：仅记录，不参与加权


def normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", s)
    return " ".join(s.split())


def self_consistency(samples: list[str]) -> float:
    """同源多次采样的一致率（多数答案占比）。单次采样返回 0.5（无信息）。"""
    vals = [normalize(s) for s in samples if s]
    if len(vals) < 2:
        return 0.5
    _, n = Counter(vals).most_common(1)[0]
    return n / len(vals)


def verdict(sources: list[Source], *, low_consistency: float = 0.5,
            weak_support: float = 0.6) -> dict:
    """给出结论、风险等级与建议。

    返回 dict：
      answer        多数来源支持的答案（无法判定时为 None）
      support       支持该答案的来源占比
      consistency   各来源自一致性（按来源名）
      risk          "low" / "medium" / "high"
      warnings      需要注意的问题
      advice        下一步该做什么
    """
    warnings: list[str] = []
    consistency = {s.name: round(self_consistency(s.samples), 2) for s in sources}

    indep = [s for s in sources if s.independent]
    shared = [s for s in sources if not s.independent]
    if shared:
        warnings.append(
            f"{len(shared)} 个来源被标记为共享输入（{', '.join(s.name for s in shared)}）："
            "实测显示同源错误高度相关，这些来源之间不构成冗余"
        )
    if len(indep) < 2:
        warnings.append(f"仅 {len(indep)} 个独立来源，投票/冗余意义有限")

    # 每个来源取自身多数答案，再跨来源汇总（与实测中的裁决策略一致）
    votes: Counter = Counter()
    supporters: dict[str, list[str]] = {}
    for s in sources:
        vals = [normalize(x) for x in s.samples if x]
        if not vals:
            continue
        top = Counter(vals).most_common(1)[0][0]
        votes[top] += 1
        supporters.setdefault(top, []).append(s.name)

    if not votes:
        return {"answer": None, "support": 0.0, "consistency": consistency,
                "risk": "high", "warnings": warnings + ["没有任何有效答案"],
                "advice": "先去取数据，再谈裁决"}

    answer, n_support = votes.most_common(1)[0]
    support = n_support / sum(votes.values())

    avg_cons = sum(consistency.values()) / len(consistency) if consistency else 0.5
    weak_sources = [k for k, v in consistency.items() if v < low_consistency]
    if weak_sources:
        warnings.append(f"自一致性偏低的来源：{', '.join(weak_sources)}（同源多次答案不一致，说明该来源对此问题无把握）")

    if support < weak_support or avg_cons < low_consistency:
        risk = "high"
        advice = ("结论可信度低：不要去加更多同类模型，改为升级证据 —— 检索一手资料、"
                  "跑工具/代码验证、或直接向人确认")
    elif warnings:
        risk = "medium"
        advice = "结论勉强可用，但存在结构性风险（见 warnings），重要场合仍需人工复核"
    else:
        risk = "low"
        advice = "多来源一致且各自稳健，可作为高可信结论记录"

    return {"answer": answer, "support": round(support, 2), "consistency": consistency,
            "risk": risk, "warnings": warnings, "advice": advice}


def selftest() -> None:
    r = verdict([Source("a", ["x", "x", "x"]), Source("b", ["x", "x", "y"])])
    assert r["answer"] == "x" and r["risk"] == "low", r
    r2 = verdict([Source("a", ["x", "y", "z"]), Source("b", ["p", "q", "r"])])
    assert r2["risk"] == "high" and "检索" in r2["advice"], r2
    r3 = verdict([Source("a", ["x", "x"]), Source("b", ["x", "x"], independent=False)])
    assert any("共享输入" in w for w in r3["warnings"]), r3
    assert self_consistency(["x"]) == 0.5
    print("selftest ok")


if __name__ == "__main__":
    selftest()
