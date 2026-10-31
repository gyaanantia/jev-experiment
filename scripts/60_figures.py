"""Slide-ready figures (16:9, PNG at 2x + SVG, light and dark) from report/data.json."""
import json

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from jevmatch.run import ROOT  # noqa: E402

D = json.loads((ROOT / "report" / "data.json").read_text())
L = D["labels"]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
THEMES = {
    "light": dict(bg="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  jev="#2a78d6", haiku="#eb6834", val="#1baf7a", paper="#52514e"),
    "dark": dict(bg="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 jev="#3987e5", haiku="#d95926", val="#199e70", paper="#c3c2b7"),
}
SHORT = {"cupid": "CU", "similarity_flooding": "SF", "coma_schema": "COS", "jev_schema": "Jev-S",
         "distribution_based": "DB", "jaccard_levenshtein": "JL", "coma_instance": "COI", "jev_instance": "Jev-I",
         "jev_hybrid": "Jev-H", "haiku_hybrid": "Haiku"}
ORDER = [("schema", ["cupid", "similarity_flooding", "coma_schema", "jev_schema"]),
         ("instance", ["distribution_based", "jaccard_levenshtein", "coma_instance", "jev_instance"]),
         ("hybrid", ["jev_hybrid", "haiku_hybrid"])]
SCEN = ["Unionable", "View-Unionable", "Joinable", "Semantically-Joinable"]
COV = D["coverage"]
HAIKU_N = COV["haiku_hybrid"]["total"]
BASELINE_LIMIT_S = D["baseline_limit_s"]  # timeout_s in scripts/32_full_baselines.py
# baselines that lost pairs to the time limit: ok < total in data.json -> coverage
TIMED_OUT = {m: c for m, c in COV.items() if not m.startswith(("jev", "haiku")) and c["ok"] < c["total"]}


def cov_tag(m):
    """Short qualifier for an all-pairs mean ("" when the method finished every non-dev pair)."""
    if m.startswith("haiku"):
        return f"{HAIKU_N}-pair subset"
    c = TIMED_OUT.get(m)
    return f"{c['ok']} of {c['total']} pairs; {c['total'] - c['ok']} timed out" if c else ""


def sample_note():
    """How much the 1,000-row sample costs the instance matchers (dev pairs where the all-row run finished)."""
    gains = [r["all_rows"] - r["sample_1000"] for r in D.get("full_rows_dev", {}).values()]
    n = D["frozen"]["baselines"]["instance_sample_size"]
    return (f"Valentine matchers read a {n:,}-row sample; with all rows, the instance matchers gained "
            f"{min(gains):.2f} to {max(gains):.2f} Recall@GT on dev pairs." if gains else "")


def timeout_note(time_bound=False):
    """One footnote sentence per method that lost pairs to the time limit."""
    return " ".join(f"{L[m]}: {c['total'] - c['ok']} of {c['total']} non-dev pairs stopped at the {BASELINE_LIMIT_S} s "
                    f"per-run limit and are left out ({c['ok']} used)" + ("; its mean time is a lower bound." if time_bound else ".")
                    for m, c in TIMED_OUT.items())


def secs(v):
    """Same rounding as the web report: 0.85 s, 3.9 s, 16 s, 4,818 s."""
    return f"{v:.2f} s" if v < 1 else f"{v:.1f} s" if v < 10 else f"{round(v):,} s"


def sec_ticks(ax):
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,g} s"))


def legend(target, T, items, **kw):
    """Legend in ink colors; items = [(label, color, kind)] with kind "box" or "dot"."""
    h = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=9) if k == "dot" else
         plt.Rectangle((0, 0), 1, 1, color=c) for _, c, k in items]
    leg = target.legend(h, [lab for lab, _, _ in items], frameon=False, fontsize=12, **kw)
    for tx in leg.get_texts():
        tx.set_color(T["ink2"])
    return leg


def footnote(f, T, text):
    f.text(0.04, 0.025, text, color=T["ink2"], fontsize=11, va="bottom", wrap=True)


def color(m, T):
    return T["jev"] if m.startswith("jev") else T["haiku"] if m.startswith("haiku") else T["val"]


def style(ax, T):
    ax.set_facecolor(T["bg"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(T["axis"])
    ax.tick_params(colors=T["ink2"], labelsize=13)
    ax.grid(axis="y", color=T["grid"], linewidth=1)
    ax.set_axisbelow(True)


def fig(T, title, sub=None):
    f = plt.figure(figsize=(13.333, 7.5), facecolor=T["bg"])
    f.text(0.04, 0.94, title, color=T["ink"], fontsize=24, fontweight="bold", va="top")
    if sub:
        f.text(0.04, 0.885, sub, color=T["ink2"], fontsize=14, va="top")
    return f


def save(f, name, theme):
    for ext in ("png", "svg"):
        f.savefig(OUT / f"{name}_{theme}.{ext}", dpi=150 if ext == "png" else None, facecolor=f.get_facecolor())
    plt.close(f)


def accuracy(T, theme):
    hn = sorted({D["accuracy"][sc]["haiku_hybrid"]["all"]["n"] for sc in SCEN if "haiku_hybrid" in D["accuracy"].get(sc, {})})
    f = fig(T, "Recall@ground truth on generated test pairs",
            "TPC-DI, Open Data, ChEMBL · box = middle 50%, line = median, dot = mean, small dots = outliers")
    legend(f, T, [("Jev", T["jev"], "box"), (f"Haiku 5.5 ({HAIKU_N}-pair subset; {'/'.join(map(str, hn))} generated pairs per scenario)", T["haiku"], "box"),
                  ("Valentine matchers, re-run", T["val"], "box")], loc="upper left", bbox_to_anchor=(0.035, 0.865), ncol=3)
    axes = f.subplots(1, 4, sharey=True, gridspec_kw=dict(left=0.05, right=0.98, top=0.74, bottom=0.16, wspace=0.08))
    for ax, sc in zip(axes, SCEN):
        style(ax, T)
        pos, x = [], 0
        for fam, ms in ORDER:
            for m in ms:
                B = D["accuracy"].get(sc, {}).get(m, {}).get("all")
                if B:
                    c = color(m, T)
                    ax.plot([x, x], [B["lo"], B["hi"]], color=c, lw=1.5)
                    ax.add_patch(plt.Rectangle((x - 0.3, B["q1"]), 0.6, max(0.01, B["q3"] - B["q1"]), facecolor=c,
                                               alpha=0.25, edgecolor=c, lw=1.5))
                    ax.plot([x - 0.3, x + 0.3], [B["median"]] * 2, color=c, lw=3)
                    ax.plot(x, B["mean"], "o", color=c, markersize=7, markeredgecolor=T["bg"], markeredgewidth=2)
                    # outliers (beyond 1.5 x IQR): without them a mean below the whisker looks like an error
                    if B.get("outliers"):
                        ax.plot([x] * len(B["outliers"]), B["outliers"], "o", color=c, markersize=3, alpha=0.6, ls="")
                pos.append((x, SHORT[m]))
                x += 1
            x += 0.6
        ax.set_xticks([p for p, _ in pos], [l for _, l in pos], rotation=60, fontsize=11)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(sc, color=T["ink"], fontsize=15, pad=10)
    footnote(f, T, timeout_note() + "\n" + sample_note())
    save(f, "01_accuracy_boxes", theme)


def scatter(T, theme):
    C = D["cost_latency"]
    f = fig(T, "Accuracy vs time per table pair",
            f"Mean Recall@GT over non-dev pairs · bars = 95% CI · log time axis · Haiku not shown ({HAIKU_N}-pair subset only)")
    ax = f.add_axes([0.08, 0.16, 0.88, 0.66])
    xmax = max(r["wall_ci"][1] for m, r in C.items() if not m.startswith("haiku"))
    style(ax, T)
    ax.grid(axis="x", color=T["grid"], linewidth=1)
    for m, r in C.items():
        if m.startswith("haiku"):
            continue
        c = color(m, T)
        ax.errorbar(r["wall_mean"], r["recall_mean"], yerr=[[r["recall_mean"] - r["recall_ci"][0]], [r["recall_ci"][1] - r["recall_mean"]]],
                    xerr=[[r["wall_mean"] - r["wall_ci"][0]], [r["wall_ci"][1] - r["wall_mean"]]], fmt="o", color=c,
                    markersize=9, markeredgecolor=T["bg"], markeredgewidth=2, elinewidth=1.5, capsize=0)
        # label to the left of points in the right third, so long names are not cut at the axes edge
        right = np.log10(r["wall_mean"]) > np.log10(xmax) - 0.6
        lab = L[m] + (f" ({cov_tag(m)})" if cov_tag(m) else "")
        ax.annotate(lab, (r["wall_mean"], r["recall_mean"]), xytext=(-8 if right else 8, 6), textcoords="offset points",
                    color=T["ink2"], fontsize=12, ha="right" if right else "left")
    ax.set_xscale("log")
    sec_ticks(ax)
    legend(ax, T, [("Jev", T["jev"], "dot"), ("Valentine matchers, re-run", T["val"], "dot")], loc="upper right")
    footnote(f, T, timeout_note(time_bound=True) + "\n" + sample_note())
    ax.set_xlabel("wall-clock seconds per table pair (this Mac)", color=T["ink2"], fontsize=13)
    ax.set_ylabel("mean Recall@GT", color=T["ink2"], fontsize=13)
    save(f, "02_accuracy_vs_time", theme)


def runtime(T, theme):
    P, C = D["paper"]["table_v"]["seconds"], D["cost_latency"]
    ms = ["coma_schema", "similarity_flooding", "cupid", "distribution_based", "coma_instance", "jaccard_levenshtein",
          "semprop", "embdi", "jev_hybrid", "haiku_hybrid"]
    iss = D["frozen"]["baselines"]["instance_sample_size"]
    f = fig(T, "Seconds per table pair: paper's Table V vs this run",
            f"Paper: two 80-core VMs, all rows · This run: one laptop, {iss:,}-row sample · log scale")
    legend(f, T, [("Paper, Table V", T["paper"], "box"), ("Valentine, this run", T["val"], "box"), ("Jev, this run", T["jev"], "box"),
                  (f"Haiku 5.5, this run ({HAIKU_N}-pair subset)", T["haiku"], "box")],
           loc="upper left", bbox_to_anchor=(0.035, 0.865), ncol=4)
    ax = f.add_axes([0.2, 0.14, 0.75, 0.66])
    style(ax, T)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=T["grid"], linewidth=1)
    y = np.arange(len(ms))[::-1]
    for yi, m in zip(y, ms):
        if m in P:
            ax.barh(yi + 0.18, P[m], height=0.32, color=T["paper"])
            ax.text(P[m] * 1.1, yi + 0.18, secs(P[m]), va="center", color=T["ink2"], fontsize=11)
        if m in C:
            v = C[m]["wall_mean"]
            ax.barh(yi - 0.18, v, height=0.32, color=color(m, T))
            q = cov_tag(m)
            ax.text(v * 1.1, yi - 0.18, secs(v) + (f" ({q})" if q else ""), va="center", color=T["ink2"], fontsize=11)
    ax.set_yticks(y, [L[m] for m in ms], fontsize=13)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 20000)
    sec_ticks(ax)
    footnote(f, T, timeout_note(time_bound=True))
    save(f, "03_runtime_vs_table_v", theme)


def sweep(T, theme):
    S = D["dev"]["sweep"]
    st = list(D["dev"]["stability"]["per_run_mean"].values())
    f = fig(T, "More sample values did not help",
            f"Jev hybrid on the 24 dev pairs · wording W2 · one run per N "
            f"(three identical runs of N = 10 spread {max(st) - min(st):.3f})")
    ax = f.add_axes([0.1, 0.12, 0.85, 0.7])
    style(ax, T)
    x = [r["tok_per_cell"] for r in S]
    y = [r["recall"] for r in S]
    ax.plot(x, y, color=T["jev"], lw=2)
    ax.plot(x, y, "o", color=T["jev"], markersize=10, markeredgecolor=T["bg"], markeredgewidth=2)
    for i, r in enumerate(S):
        # label on the side away from the line: below a point the line rises from, above otherwise
        rising = i + 1 < len(S) and S[i + 1]["recall"] > r["recall"]
        ax.annotate(f"N = {r['N']}", (r["tok_per_cell"], r["recall"]), xytext=(0, -22 if rising else 12),
                    textcoords="offset points", color=T["ink2"], fontsize=13, ha="center")
    lo = min(y)
    assert lo > 0.97, "a sweep point falls below the 0.97 axis floor"
    ax.set_xlabel("input tokens per column pair", color=T["ink2"], fontsize=13)
    ax.set_ylabel("mean Recall@GT (axis starts at 0.97)", color=T["ink2"], fontsize=13)
    ax.set_ylim(0.97, 1.0)
    save(f, "04_sample_size_sweep", theme)


def cost(T, theme):
    C = D["cost_latency"]
    ms = [m for m in ["jev_schema", "jev_instance", "jev_hybrid", "haiku_hybrid"] if m in C]
    f = fig(T, "API cost per table pair", f"Mean over non-dev pairs (Haiku: {HAIKU_N}-pair subset) · Valentine matchers: $0 API, local CPU")
    ax = f.add_axes([0.2, 0.12, 0.72, 0.68])
    style(ax, T)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=T["grid"], linewidth=1)
    for i, m in enumerate(ms[::-1]):
        v = C[m]["cost_mean"]
        ax.barh(i, v, height=0.45, color=color(m, T))
        ax.annotate(f"${v:.4f}", (v, i), xytext=(8, 0), textcoords="offset points", va="center", color=T["ink2"], fontsize=13)
    ax.set_yticks(range(len(ms)), [L[m] + (f"\n({cov_tag(m)})" if m.startswith("haiku") else "") for m in ms[::-1]], fontsize=14)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:.3f}"))
    legend(ax, T, [("Jev", T["jev"], "box"), (f"Haiku 5.5 ({HAIKU_N}-pair subset)", T["haiku"], "box")], loc="upper right")
    save(f, "05_cost_per_pair", theme)


def main():
    for theme, T in THEMES.items():
        plt.rcParams.update({"font.family": "sans-serif", "svg.fonttype": "none"})
        accuracy(T, theme)
        scatter(T, theme)
        runtime(T, theme)
        sweep(T, theme)
        cost(T, theme)
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
