# -*- coding: utf-8 -*-
"""异构多智能体共识实验 v2c —— 只用实测可用的渠道，靠加深采样提升统计质量。

背景：v2b 跑下来发现 OpenRouter 免费池额度耗尽（全线 429，0/198 成功）、
中转站 Groq 渠道部分限流。实测 100% 可用的只有直连的 DeepSeek 与智谱。

因此本版：
  * 只保留 DeepSeek(10 次采样) / Zhipu(5) / GroqCompound(3)
  * DeepSeek 的 10 次采样同时充当「同质组」基线，用于对比「同一模型多次采样」
    与「不同厂商模型」的错误相关性差异
  * 断点续跑只跳过**成功**记录，失败的允许重试
"""
import csv
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

DSH = os.path.expanduser(r"~\.dsh")
KEYS = json.load(open(os.path.join(DSH, "llm-keys.json"), encoding="utf-8"))["entries"]
NEWAPI = open(os.path.join(DSH, "newapi.token"), encoding="utf-8-sig").read().strip().lstrip("\ufeff")
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("HETERO_DATA", os.path.join(os.path.dirname(HERE), "data"))
RAW = os.path.join(OUTDIR, "raw.jsonl")
os.makedirs(OUTDIR, exist_ok=True)

N_QUESTIONS = 30
SEED = 20260911
RELAY = os.environ.get("HETERO_RELAY", "http://127.0.0.1:8899/v1")  # 自建 OpenAI 兼容中转；请用环境变量覆盖

# (名称, 模型, 端点, key, 限流桶, 采样次数)
PROVIDERS = [
    ("DeepSeek", "deepseek-chat", "https://api.deepseek.com/chat/completions", KEYS.get("deepseek"), "fast", 10),
    ("Zhipu", "glm-4-flash", "https://open.bigmodel.cn/api/paas/v4/chat/completions", KEYS.get("智谱4.7"), "fast", 5),
    ("GroqCompound", "groq/compound-mini", f"{RELAY}/chat/completions", NEWAPI, "relay", 3),
]

INTERVAL = {"fast": 0.0, "relay": 0.6}
_last = defaultdict(float)
_throttle_lock = threading.Lock()
_write_lock = threading.Lock()


def throttle(bucket):
    need = INTERVAL.get(bucket, 0.0)
    if need <= 0:
        return
    while True:
        with _throttle_lock:
            now = time.time()
            wait = need - (now - _last[bucket])
            if wait <= 0:
                _last[bucket] = now
                return
        time.sleep(min(wait, 0.5))


PROMPT = """问题：{q}

要求：
1. 只写出这个问题的答案本身，尽量简短（人名、日期、数字或短语）。
2. 然后另起一行，输出你认为自己答对的把握，格式必须严格为：CONFIDENCE: <0 到 1 的小数>
3. 不要重复题目，不要输出解释或其他内容。"""


def ask(model, url, key, question, temperature=0.8, retries=2):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(q=question)}],
        "max_tokens": 700,
        "temperature": temperature,
    }).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=body, headers=headers)
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=75) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
            usage = data.get("usage") or {}
            return {"ok": True, "sec": round(time.time() - t0, 1),
                    "text": ((msg.get("content") or "") + "\n" + (msg.get("reasoning") or "")).strip(),
                    "in": usage.get("prompt_tokens"), "out": usage.get("completion_tokens")}
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:100]}"
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2.0 * (attempt + 1))
                continue
            break
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            time.sleep(1.0 * (attempt + 1))
    return {"ok": False, "error": err}


def parse_output(text):
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


def normalize(s):
    s = (s or "").lower()
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", s)
    return " ".join(s.split())


def is_correct(pred, gold):
    p, g = normalize(pred), normalize(gold)
    if not p or not g:
        return False
    if g in p or p in g:
        return True
    gn, pn = re.findall(r"\d+", g), re.findall(r"\d+", p)
    if gn and gn == pn:
        return True
    gw = [w for w in g.split() if len(w) > 2]
    return bool(gw) and all(w in p.split() for w in gw)


def load_questions():
    path = os.path.join(OUTDIR, "questions.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    rows = list(csv.DictReader(open(os.path.join(OUTDIR, "simple_qa_test_set.csv"), encoding="utf-8")))
    rng = random.Random(SEED)
    qs = []
    for i, r in enumerate(rng.sample(rows, N_QUESTIONS), 1):
        meta = {}
        try:
            meta = json.loads(r.get("metadata") or "{}")
        except Exception:  # noqa: BLE001
            pass
        qs.append({"qid": f"Q{i:02d}", "question": r["problem"], "gold": r["answer"],
                   "topic": meta.get("topic", "?"), "answer_type": meta.get("answer_type", "?")})
    json.dump(qs, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return qs


def load_done():
    """只把成功的记录当作已完成；失败项允许重试。"""
    done = set()
    if os.path.exists(RAW):
        for line in open(RAW, encoding="utf-8"):
            try:
                r = json.loads(line)
                if r.get("ok"):
                    done.add((r["vendor"], r["qid"], r["sample"]))
            except Exception:  # noqa: BLE001
                pass
    return done


def append_raw(r):
    with _write_lock:
        with open(RAW, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    qs = load_questions()
    done = load_done()
    planned = [(v, m, u, k, b) for (v, m, u, k, b, s) in PROVIDERS for _ in range(1)]
    total = sum(s for *_, s in PROVIDERS) * len(qs)
    print(f"计划 {total} 次调用，已完成 {len(done)} 条成功记录", flush=True)

    def task(vendor, model, url, key, bucket, q, s):
        throttle(bucket)
        r = ask(model, url, key, q["question"])
        r.update({"vendor": vendor, "model": model, "qid": q["qid"], "sample": s,
                  "gold": q["gold"], "topic": q["topic"]})
        if r.get("ok"):
            ans, conf = parse_output(r["text"])
            r["answer"], r["confidence"] = ans, conf
            r["correct"] = is_correct(ans, q["gold"])
        else:
            r["answer"] = r["confidence"] = None
            r["correct"] = False
        append_raw(r)
        return r

    futures = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for vendor, model, url, key, bucket, samples in PROVIDERS:
            for q in qs:
                for s in range(samples):
                    if (vendor, q["qid"], s) in done:
                        continue
                    futures.append(pool.submit(task, vendor, model, url, key, bucket, q, s))
        n = 0
        for fut in futures:
            fut.result()
            n += 1
            if n % 40 == 0:
                recs = [json.loads(l) for l in open(RAW, encoding="utf-8") if l.strip()]
                oks = [x for x in recs if x.get("ok")]
                acc = sum(1 for x in oks if x.get("correct"))
                print(f"  新增 {n}/{len(futures)}  累计成功 {len(oks)}  正确率 {acc/max(len(oks),1):.1%}", flush=True)

    recs = [json.loads(l) for l in open(RAW, encoding="utf-8") if l.strip()]
    oks = [r for r in recs if r.get("ok")]
    print(f"\n完成：记录 {len(recs)} 条，成功 {len(oks)}", flush=True)
    print("各渠道成功数:", dict(Counter(r["vendor"] for r in oks)), flush=True)


if __name__ == "__main__":
    main()
