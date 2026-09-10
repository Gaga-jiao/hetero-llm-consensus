# -*- coding: utf-8 -*-
"""真实 LLM provider 适配层 —— 让 CP-WBFT 用真实模型，而不是手工设定的置信度。

背景
----
上游 cp-wbft README 的 "Next Steps" 第一条是
``connect real LLM providers for PCP response generation``。
当前实现里信心值完全由调用方传入（``honest_confidence=0.9`` /
``byzantine_confidence=0.05``），仿真结论因此对"错误答案一定低置信度"
这个假设高度敏感。

实测发现（8 家厂商 × SimpleQA）
--------------------------------
真实模型上这个假设不成立：答错时的自报置信度与答对时几乎相同
（例如 0.96 vs 0.98）。也就是说单一的 prompt 自报置信度**饱和**，
直接拿来做加权共识等价于等权投票。

因此本模块除了解析自报置信度，还提供 ``self_consistency_confidence()``：
同一问题多次独立采样，用多数答案的出现比例作为置信度代理。这是一个可测、
不依赖模型自我报告的信号。

用法
----
>>> p = OpenAICompatProvider("deepseek", "https://api.deepseek.com", key, "deepseek-chat")
>>> r = p.query("Who received the IEEE Frank Rosenblatt Award in 2010?")
>>> r.answer, r.confidence
('Michio Sugeno', 0.99)
>>> states = states_from_providers([p, p2, p3], question)
>>> CPWBFT(build_topology("complete", len(states))).decide(states)
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass

try:  # 作为包的一部分使用时
    from .models import AgentState
except ImportError:  # 独立脚本方式使用时
    from dataclasses import dataclass as _dc

    @_dc(frozen=True)
    class AgentState:  # type: ignore[no-redef]
        agent_id: int
        answer: str
        confidence: float
        is_byzantine: bool = False


PROMPT = """问题：{q}

要求：
1. 只写出这个问题的答案本身，尽量简短（人名、日期、数字或短语）。
2. 然后另起一行，输出你认为自己答对的把握，格式必须严格为：CONFIDENCE: <0 到 1 的小数>
3. 不要重复题目，不要输出解释或其他内容。"""


@dataclass
class ProviderResponse:
    text: str
    answer: str
    confidence: float | None


class OpenAICompatProvider:
    """任意 OpenAI 兼容端点：DeepSeek / 智谱 / Groq / OpenRouter / 本地 vLLM 均可。"""

    def __init__(self, name: str, base_url: str, api_key: str, model: str, *,
                 temperature: float = 0.8, timeout: int = 75, retries: int = 2,
                 min_interval: float = 0.0):
        self.name = name
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries
        self.min_interval = min_interval
        self._last = 0.0

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()

    def query(self, question: str, temperature: float | None = None) -> ProviderResponse:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": PROMPT.format(q=question)}],
            "max_tokens": 700,
            "temperature": self.temperature if temperature is None else temperature,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                self._throttle()
                req = urllib.request.Request(self.endpoint, data=body, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
                text = ((msg.get("content") or "") + "\n" + (msg.get("reasoning") or "")).strip()
                answer, confidence = parse_pcp(text)
                return ProviderResponse(text=text, answer=answer, confidence=confidence)
            except urllib.error.HTTPError as e:
                last_err = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:120]}"
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"{self.name} 查询失败：{last_err}")


def parse_pcp(text: str) -> tuple[str, float | None]:
    """从模型输出里解析 prompt-level confidence probing (PCP) 结果。"""
    conf = None
    m = re.search(r"CONFIDENCE\s*[:：]\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"置信度\s*[:：]\s*([0-9]*\.?[0-9]+)", text)
    if m:
        try:
            conf = max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            conf = None
        body = text[:m.start()]
    else:
        body = text
    body = re.sub(r"^\s*(答案|Answer)\s*[:：]\s*", "", body.strip(), flags=re.IGNORECASE)
    return " ".join(body.split())[:200], conf


def self_consistency_confidence(provider: OpenAICompatProvider, question: str,
                                samples: int = 5) -> tuple[str, float]:
    """用多次独立采样的多数答案比例作为置信度代理。

    比 prompt 自报置信度更可靠：它是可观测量，不依赖模型对自身正确性的判断。
    返回 (多数答案, 一致率)。
    """
    answers = []
    for i in range(samples):
        try:
            answers.append(provider.query(question, temperature=max(provider.temperature, 0.8)).answer)
        except RuntimeError:
            continue
    if not answers:
        return "", 0.0
    top, n = Counter(answers).most_common(1)[0]
    return top, n / len(answers)


def states_from_providers(providers, question: str, *, samples: int = 1,
                          byzantine: set[int] | None = None) -> list[AgentState]:
    """把多个真实 provider 的作答转成 CP-WBFT 需要的 AgentState 列表。

    samples > 1 时使用自一致性作为置信度（推荐），否则用模型自报置信度。
    """
    byzantine = byzantine or set()
    states: list[AgentState] = []
    for i, p in enumerate(providers):
        if samples > 1:
            answer, confidence = self_consistency_confidence(p, question, samples)
        else:
            r = p.query(question)
            answer, confidence = r.answer, (r.confidence if r.confidence is not None else 0.5)
        states.append(AgentState(agent_id=i, answer=answer, confidence=confidence,
                                 is_byzantine=i in byzantine))
    return states
