# -*- coding: utf-8 -*-
"""异构多智能体共识实验 v2 分析。

输入：~/.dsh/hetero-lab/v2_raw.jsonl + questions.json
输出：控制台汇总 + summary.json + 图（matplotlib 可用时）
"""
import json
import math
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTDIR = os.environ.get("HETERO_DATA", os.path.join(ROOT, "data"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:  # noqa: BLE001
    HAS_MPL = False


# ---------- 统计工具 ----------
def ece(pairs, bins=10):
    """Expected Calibration Error。pairs = [(confidence, correct_bool)]"""
    pairs = [(c, bool(k)) for c, k in pairs if c is not None]
    total = len(pairs)
    if not total:
        return None, []
    e, curve = 0.0, []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        bucket = [(c, k) for c, k in pairs if (lo < c <= hi) or (i == 0 and c <= hi)]
        if not bucket:
            curve.append((round((lo + hi) / 2, 3), None, 0))
            continue
        acc = sum(1 for _, k in bucket if k) / len(bucket)
        conf = sum(c for c, _ in bucket) / len(bucket)
        e += (len(bucket) / total) * abs(acc - conf)
        curve.append((round(conf, 3), round(acc, 3), len(bucket)))
    return e, curve


def apply_temp(p, t):
    p = min(max(p, 1e-6), 1 - 1e-6)
    logit = math.log(p / (1 - p))
    return 1 / (1 + math.exp(-logit / t))


def best_temp(pairs, grid=None):
    """网格搜索温度，使 ECE 最小（仅用自报置信度，属于事后校准）。"""
    grid = grid or [0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    best = (None, None)
    for t in grid:
        e, _ = ece([(apply_temp(c, t), k) for c, k in pairs])
        if e is not None and (best[0] is None or e < best[0]):
            best = (e, t)
    return best


def phi(pairs):
    """pairs = [(bool_a, bool_b)]，返回 phi 相关系数。"""
    n11 = sum(1 for a, b in pairs if a and b)
    n10 = sum(1 for a, b in pairs if a and not b)
    n01 = sum(1 for a, b in pairs if not a and b)
    n00 = sum(1 for a, b in pairs if not a and not b)
    denom = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return ((n11 * n00 - n10 * n01) / denom) if denom else 0.0


def selftest():
    # 全对但只报 0.5 的置信度 → ECE 应为 0.5
    assert abs(ece([(0.5, True)] * 10)[0] - 0.5) < 1e-6
    # 报 0.9 且实际 90% 正确 → 校准良好，ECE ≈ 0
    assert ece([(0.9, True)] * 9 + [(0.9, False)])[0] < 1e-6
    assert abs(phi([(True, True), (False, False), (True, True), (False, False)]) - 1.0) < 1e-9
    assert abs(phi([(True, False), (False, True), (True, False), (False, True)]) + 1.0) < 1e-9
    # 温度 >1 应把极端置信度向 0.5 压缩
    assert apply_temp(0.99, 5.0) < apply_temp(0.99, 2.0) < apply_temp(0.99, 1.0) <= 0.99
    print("selftest ok")


# ---------- 载入 ----------
def load():
    recs = [json.loads(l) for l in open(os.path.join(OUTDIR, "raw.jsonl"), encoding="utf-8") if l.strip()]
    qs = {q["qid"]: q for q in json.load(open(os.path.join(OUTDIR, "questions.json"), encoding="utf-8"))}
    return recs, qs


def main():
    recs, qs = load()
    ok = [r for r in recs if r.get("ok")]
    vendors = sorted({r["vendor"] for r in ok})
    qids = sorted(qs)
    print(f"记录 {len(recs)} 条，成功 {len(ok)}，模型 {len(vendors)} 家，题目 {len(qids)} 道\n")

    # 每个 (vendor, qid) 聚合：3 次采样
    agg = defaultdict(lambda: {"answers": [], "confs": [], "corrects": []})
    for r in ok:
        a = agg[(r["vendor"], r["qid"])]
        a["answers"].append(r["answer"])
        if r["confidence"] is not None:
            a["confs"].append(r["confidence"])
        a["corrects"].append(bool(r["correct"]))

    per = {}
    for (v, qid), a in agg.items():
        cnt = Counter(x for x in a["answers"] if x)
        top, top_n = (cnt.most_common(1)[0] if cnt else (None, 0))
        per[(v, qid)] = {
            "answer": top,
            "consistency": top_n / len(a["answers"]) if a["answers"] else 0.0,
            "conf": (sum(a["confs"]) / len(a["confs"])) if a["confs"] else None,
            "correct": is_any_correct(a["answers"], qs[qid]["gold"]),
            "vote_correct": is_any_correct([top] if top else [], qs[qid]["gold"]),
            "n": len(a["answers"]),
        }

    # 【1】单模型准确率 + 置信度
    print("=" * 76)
    print("【1】单模型表现（任一采样答对 / 多数答案答对）")
    print(f"{'模型':<16}{'任一采样对':>10}{'多数答案对':>12}{'平均置信度':>12}{'平均自一致性':>14}")
    single = {}
    for v in vendors:
        rs = [per[(v, q)] for q in qids if (v, q) in per]
        if not rs:
            continue
        any_acc = sum(1 for x in rs if x["correct"]) / len(rs)
        vote_acc = sum(1 for x in rs if x["vote_correct"]) / len(rs)
        confs = [x["conf"] for x in rs if x["conf"] is not None]
        cons = [x["consistency"] for x in rs]
        single[v] = {"any": any_acc, "vote": vote_acc,
                     "conf": sum(confs) / len(confs) if confs else None,
                     "consistency": sum(cons) / len(cons)}
        print(f"{v:<16}{any_acc:>10.0%}{vote_acc:>12.0%}"
              f"{(single[v]['conf'] or 0):>12.3f}{single[v]['consistency']:>14.3f}")

    # 【2】置信度校准
    print("\n" + "=" * 76)
    print("【2】置信度校准（ECE 越低越准；温度缩放为事后校准）")
    pairs = []
    for v in vendors:
        for q in qids:
            x = per.get((v, q))
            if x and x["conf"] is not None:
                pairs.append((x["conf"], x["vote_correct"]))
    e_raw, curve = ece(pairs)
    e_best, t_best = best_temp(pairs)
    print(f"样本 {len(pairs)} 条   原始 ECE = {e_raw:.3f}   最优温度 T = {t_best} → ECE = {e_best:.3f}（降幅 {(1-e_best/e_raw)*100:.0f}%）")
    print(f"{'置信度区间':>12}{'样本数':>8}{'平均置信度':>12}{'实际准确率':>12}{'差距':>10}")
    for conf, acc, n in curve:
        if acc is None:
            continue
        print(f"{conf:>12.2f}{n:>8}{conf:>12.3f}{acc:>12.3f}{acc-conf:>10.3f}")

    # 【3】自一致性 vs 正确率
    print("\n" + "=" * 76)
    print("【3】自一致性（3 次采样多数答案占比）vs 实际正确率")
    buckets = defaultdict(lambda: [0, 0])
    for v in vendors:
        for q in qids:
            x = per.get((v, q))
            if not x:
                continue
            key = round(x["consistency"], 2)
            buckets[key][0] += 1
            buckets[key][1] += 1 if x["vote_correct"] else 0
    for key in sorted(buckets):
        n, c = buckets[key]
        print(f"  一致性 {key:.2f}  样本 {n:>4}  正确率 {c/n:>6.1%}")

    # 【4】错误相关性
    print("\n" + "=" * 76)
    print("【4】跨模型正确性相关性（phi 系数：正=一起对/一起错，负=互补）")
    mat = {}
    for i, a in enumerate(vendors):
        for b in vendors[i + 1:]:
            pr = [(per[(a, q)]["vote_correct"], per[(b, q)]["vote_correct"])
                  for q in qids if (a, q) in per and (b, q) in per]
            if pr:
                mat[(a, b)] = phi(pr)
    for (a, b), val in sorted(mat.items(), key=lambda kv: -kv[1]):
        print(f"  {a:<14} × {b:<14} phi = {val:+.2f}")
    if mat:
        print(f"  平均 phi = {sum(mat.values())/len(mat):+.2f}")

    # 【4b】同质组：同一模型多次采样之间的相关性（核心对照）
    print("\n" + "=" * 76)
    print("【4b】同质组：同一模型（DeepSeek）多次采样之间的正确性相关性")
    ds = defaultdict(dict)
    for r in ok:
        if r["vendor"] == "DeepSeek":
            ds[r["qid"]][r["sample"]] = bool(r["correct"])
    smp = sorted({s for q in ds.values() for s in q})
    homo = []
    for i, a in enumerate(smp):
        for b in smp[i + 1:]:
            pr = [(ds[q][a], ds[q][b]) for q in ds if a in ds[q] and b in ds[q]]
            if pr:
                homo.append(phi(pr))
    he = (sum(mat.values()) / len(mat)) if mat else None
    ho = (sum(homo) / len(homo)) if homo else None
    if ho is not None:
        print(f"  采样数 {len(smp)}，两两组合 {len(homo)} 对，平均 phi = {ho:+.2f}")
    if he is not None and ho is not None:
        print(f"  异质（不同厂商）平均 phi = {he:+.2f}    同质（同模型）平均 phi = {ho:+.2f}")
        print(f"  → 差值 {he - ho:+.2f}：越接近 0 说明「换厂商」并没有换来额外的错误独立性")

    # 【5】四种裁决策略
    print("\n" + "=" * 76)
    print("【5】裁决策略对比（每题：各模型用其多数答案参与裁决）")
    strategies = {"majority": [], "conf_weighted": [], "consistency_weighted": [], "calibrated_weighted": []}
    for q in qids:
        cand = [(v, per[(v, q)]) for v in vendors if (v, q) in per and per[(v, q)]["answer"]]
        if not cand:
            continue
        for name in strategies:
            weights = {}
            for v, x in cand:
                w = 1.0
                if name == "conf_weighted":
                    w = x["conf"] if x["conf"] is not None else 0.5
                elif name == "consistency_weighted":
                    w = x["consistency"]
                elif name == "calibrated_weighted":
                    w = apply_temp(x["conf"], t_best) if x["conf"] is not None else 0.5
                weights[x["answer"]] = weights.get(x["answer"], 0.0) + w
            win = max(weights.items(), key=lambda kv: kv[1])[0]
            strategies[name].append(is_any_correct([win], qs[q]["gold"]))
    best_single = max((s["vote"] for s in single.values()), default=0)
    print(f"{'策略':<24}{'准确率':>10}")
    print(f"{'最好单模型':<24}{best_single:>10.0%}")
    for name, res in strategies.items():
        print(f"{name:<24}{sum(res)/len(res):>10.0%}" if res else f"{name:<24}{'-':>10}")

    # 存 summary
    summary = {
        "n_records": len(recs), "n_ok": len(ok), "vendors": vendors, "n_questions": len(qids),
        "single": single, "ece_raw": e_raw, "ece_calibrated": e_best, "best_temp": t_best,
        "calibration_curve": curve, "phi": {f"{a}|{b}": v for (a, b), v in mat.items()},
        "strategies": {k: (sum(v) / len(v) if v else None) for k, v in strategies.items()},
        "best_single": best_single,
    }
    json.dump(summary, open(os.path.join(ROOT, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n汇总已存: {os.path.join(OUTDIR, 'summary.json')}")

    # 图
    if HAS_MPL:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        cs = [(c, k) for c, k in pairs]
        xs = [c for c, _ in cs]
        ys = [1 if k else 0 for _, k in cs]
        axes[0].plot([0, 1], [0, 1], "k--", lw=1)
        axes[0].scatter(xs, ys, alpha=0.15, s=12)
        axes[0].set_title("Verbal confidence vs correctness")
        axes[0].set_xlabel("self-reported confidence")
        axes[0].set_ylabel("correct (1/0)")

        cc = [(c, a) for c, a, n in curve if a is not None]
        if cc:
            axes[1].plot([0, 1], [0, 1], "k--", lw=1)
            axes[1].plot([x[0] for x in cc], [x[1] for x in cc], "o-")
            axes[1].set_title(f"Reliability diagram (ECE={e_raw:.3f})")
            axes[1].set_xlabel("mean confidence")
            axes[1].set_ylabel("accuracy")

        names = ["best_single"] + list(strategies)
        vals = [best_single] + [sum(v) / len(v) if v else 0 for v in strategies.values()]
        axes[2].bar(range(len(names)), vals)
        axes[2].set_xticks(range(len(names)))
        axes[2].set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
        axes[2].set_ylim(0, 1)
        axes[2].set_title("Adjudication strategies")
        plt.tight_layout()
        fig_path = os.path.join(ROOT, "figures", "fig1_overview.png")
        plt.savefig(fig_path, dpi=150)
        print(f"图已存: {fig_path}")
    else:
        print("（未安装 matplotlib，跳过出图）")


def is_any_correct(answers, gold):
    import re
    def norm(s):
        s = (s or "").lower()
        s = re.sub(r"(\d),(\d)", r"\1\2", s)
        s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", s)
        return " ".join(s.split())
    g = norm(gold)
    for a in answers:
        p = norm(a)
        if not p or not g:
            continue
        if g in p or p in g:
            return True
        if re.findall(r"\d+", g) and re.findall(r"\d+", g) == re.findall(r"\d+", p):
            return True
        gw = [w for w in g.split() if len(w) > 2]
        if gw and all(w in p.split() for w in gw):
            return True
    return False


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        selftest()
    else:
        main()
