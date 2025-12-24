import numpy as np
import pandas as pd
from pathlib import Path
import time

def compute_f_stat(groups):
    arrays = [np.asarray(g) for g in groups]
    ns = np.array([a.size for a in arrays])
    means = np.array([a.mean() for a in arrays])
    grand_mean = np.sum(ns * means) / ns.sum()
    ss_between = np.sum(ns * (means - grand_mean) ** 2)
    ss_within = np.sum([np.sum((a - m) ** 2) for a, m in zip(arrays, means)])
    k = len(arrays)
    N = ns.sum()
    ms_between = ss_between / (k - 1) if k > 1 else 0.0
    ms_within = ss_within / (N - k) if N - k > 0 else 0.0
    if ms_within == 0:
        return np.inf
    return ms_between / ms_within


def permutation_test_f(groups, n_permutations=5000, seed=0):
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(g) for g in groups]
    sizes = [a.size for a in arrays]
    pooled = np.concatenate(arrays)
    observed = compute_f_stat(arrays)
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(pooled)
        start = 0
        perm_groups = []
        for s in sizes:
            perm_groups.append(pooled[start:start + s])
            start += s
        f = compute_f_stat(perm_groups)
        if f >= observed:
            count += 1
    p_value = (count + 1) / (n_permutations + 1)
    return observed, p_value


def pairwise_permutation(a, b, n_permutations=5000, seed=1):
    rng = np.random.default_rng(seed)
    a = np.asarray(a)
    b = np.asarray(b)
    obs_diff = np.abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(pooled)
        na = a.size
        pa = pooled[:na]
        pb = pooled[na:]
        if np.abs(pa.mean() - pb.mean()) >= obs_diff:
            count += 1
    p = (count + 1) / (n_permutations + 1)
    return obs_diff, p


def main():
    data_dir = Path("data")
    merged_fp = data_dir / "merged_by_date.csv"
    out_fp = data_dir / "stat_tests.txt"
    if not merged_fp.exists():
        print("Missing merged_by_date.csv")
        return

    df = pd.read_csv(merged_fp, parse_dates=["date"]) 
    if "classification" not in df.columns or "total_pnl" not in df.columns:
        print("Required columns missing in merged file")
        return

    df = df.dropna(subset=["classification", "total_pnl"]).copy()
    groups = [g["total_pnl"].values for _, g in df.groupby("classification")]
    names = [k for k, _ in df.groupby("classification")]

    lines = []
    lines.append(f"Statistical tests run at {time.ctime()}\n")
    lines.append(f"Groups: {list(names)}\n")
    for n, g in zip(names, groups):
        lines.append(f"{n}: n={len(g)}, mean={np.mean(g):.3f}, median={np.median(g):.3f}\n")

    # Try SciPy kruskal if available
    try:
        from scipy import stats
        kw_stat, kw_p = stats.kruskal(*groups)
        lines.append(f"Kruskal-Wallis H={kw_stat:.4f}, p={kw_p:.4g} (scipy.stats)\n")
    except Exception:
        lines.append("scipy not available — running permutation F-test fallback\n")
        obs_f, p_val = permutation_test_f(groups, n_permutations=5000, seed=42)
        lines.append(f"Permutation F-statistic={obs_f:.4f}, p={p_val:.4g}\n")

    # Pairwise tests (permutation)
    lines.append("\nPairwise permutation tests (mean difference):\n")
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            obsd, p = pairwise_permutation(groups[i], groups[j], n_permutations=5000, seed=100 + i + j)
            lines.append(f"{names[i]} vs {names[j]}: mean_diff={obsd:.3f}, p={p:.4g}\n")

    out_fp.write_text("".join(lines))
    print("Wrote", out_fp)


if __name__ == "__main__":
    main()
