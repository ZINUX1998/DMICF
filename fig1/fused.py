import argparse
import json
import os
import pickle
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
from scipy.stats import ks_2samp, spearmanr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perspective-dependence analysis for implicit-feedback recommendation datasets."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="Data/tmall",
        help="Directory containing trnMat.pkl, tstMat.pkl, uu10_csr, ii10_csr.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Top-k local averaging used in the affinity definition.",
    )
    parser.add_argument(
        "--num_pos",
        type=int,
        default=20000,
        help="Number of positive test interactions used for visualization.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="analysis_outputs/perspective_dependence_tmall",
        help="Output directory for csv/pdf/json artifacts.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Train split used in the late-fusion diagnostic.",
    )
    return parser.parse_args()


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_binary_csr(mat):
    if not sp.isspmatrix_csr(mat):
        mat = mat.tocsr()
    if mat.dtype != np.float32:
        mat = mat.astype(np.float32)
    mat = (mat != 0).astype(np.float32).tocsr()
    mat.sort_indices()
    return mat


def get_histories_from_csr(csr_mat):
    hist = {}
    indptr = csr_mat.indptr
    indices = csr_mat.indices
    for row in range(csr_mat.shape[0]):
        hist[row] = indices[indptr[row] : indptr[row + 1]].tolist()
    return hist


def topk_mean_sparse_row(sim_csr, row_id, candidate_ids, topk):
    if len(candidate_ids) == 0:
        return 0.0
    row = sim_csr.getrow(row_id)
    vals = row[:, np.asarray(candidate_ids, dtype=np.int64)].toarray().ravel()
    vals = vals[vals > 0]
    if vals.size == 0:
        return 0.0
    if vals.size > topk:
        vals = np.partition(vals, -topk)[-topk:]
    return float(vals.mean())


def build_degree_deciles(degrees):
    degree_rank = degrees.rank(method="first")
    deciles = pd.qcut(degree_rank, 10, labels=False, duplicates="drop")
    return deciles.astype(int)


def sample_matched_negative(user_id, pos_item, item_decile, items_by_decile, observed_items, rng):
    dec = int(item_decile[pos_item])
    pool = items_by_decile[dec]
    candidates = [it for it in pool if it not in observed_items]
    if not candidates:
        return None
    return int(rng.choice(candidates))


def infer_dataset_tag(data_dir):
    norm = os.path.normpath(data_dir)
    base = os.path.basename(norm)
    return base.replace("-", "").replace("_", "").lower()


def compute_affinity_dataframe(train_csr, test_csr, uu_csr, ii_csr, topk, num_pos, seed):
    rng = np.random.default_rng(seed)

    num_users, num_items = train_csr.shape

    train_user_hist = get_histories_from_csr(train_csr)
    train_item_hist = get_histories_from_csr(train_csr.transpose().tocsr())
    test_user_hist = get_histories_from_csr(test_csr)

    observed_by_user = {}
    for u in range(num_users):
        observed_by_user[u] = set(train_user_hist.get(u, [])) | set(test_user_hist.get(u, []))

    user_deg = np.asarray(train_csr.sum(axis=1)).reshape(-1)
    item_deg = np.asarray(train_csr.sum(axis=0)).reshape(-1)

    user_deg_s = pd.Series(user_deg)
    item_deg_s = pd.Series(item_deg)
    user_decile = build_degree_deciles(user_deg_s)
    item_decile = build_degree_deciles(item_deg_s)

    items_by_decile = defaultdict(list)
    for item_id, dec in item_decile.items():
        items_by_decile[int(dec)].append(int(item_id))

    test_rows, test_cols = test_csr.nonzero()
    pos_df = pd.DataFrame({"user": test_rows, "item": test_cols})
    pos_df["user_deg_decile"] = pos_df["user"].map(user_decile)

    quota = max(1, num_pos // max(1, pos_df["user_deg_decile"].nunique()))
    sampled_parts = []
    for _, grp in pos_df.groupby("user_deg_decile"):
        take = min(quota, len(grp))
        sampled_parts.append(grp.sample(n=take, random_state=seed))
    vis_pos = pd.concat(sampled_parts, axis=0)
    if len(vis_pos) > num_pos:
        vis_pos = vis_pos.sample(n=num_pos, random_state=seed)

    rows = []
    skipped = 0

    pair_id = 0

    for _, row in vis_pos.iterrows():
        u = int(row["user"])
        j_pos = int(row["item"])

        a_u_pos = topk_mean_sparse_row(ii_csr, j_pos, train_user_hist[u], topk)
        a_v_pos = topk_mean_sparse_row(uu_csr, u, train_item_hist[j_pos], topk)

        rows.append(
            {
                "user": u,
                "item": j_pos,
                "label": "Positive",
                "alpha_user": a_u_pos,
                "alpha_item": a_v_pos,
                "user_deg": float(user_deg[u]),
                "item_deg": float(item_deg[j_pos]),
                "user_deg_decile": int(user_decile[u]),
                "item_deg_decile": int(item_decile[j_pos]),
                "pair_id": int(pair_id),
            }
        )

        j_neg = sample_matched_negative(
            user_id=u,
            pos_item=j_pos,
            item_decile=item_decile,
            items_by_decile=items_by_decile,
            observed_items=observed_by_user[u],
            rng=rng,
        )
        if j_neg is None:
            skipped += 1
            continue

        a_u_neg = topk_mean_sparse_row(ii_csr, j_neg, train_user_hist[u], topk)
        a_v_neg = topk_mean_sparse_row(uu_csr, u, train_item_hist[j_neg], topk)

        rows.append(
            {
                "user": u,
                "item": j_neg,
                "label": "Matched Negative",
                "alpha_user": a_u_neg,
                "alpha_item": a_v_neg,
                "user_deg": float(user_deg[u]),
                "item_deg": float(item_deg[j_neg]),
                "user_deg_decile": int(user_decile[u]),
                "item_deg_decile": int(item_decile[j_neg]),
                "pair_id": int(pair_id),
            }
        )
        pair_id += 1

    affinity_df = pd.DataFrame(rows)
    return affinity_df, skipped


def compute_summary_stats(df):
    pos = df[df["label"] == "Positive"]
    neg = df[df["label"] == "Matched Negative"]

    def safe_spearman(x, y):
        if len(x) < 2:
            return np.nan
        return float(spearmanr(x, y).statistic)

    summary = {
        "num_positive": int(len(pos)),
        "num_negative": int(len(neg)),
        "alpha_user_mean_positive": float(pos["alpha_user"].mean()),
        "alpha_user_mean_negative": float(neg["alpha_user"].mean()),
        "alpha_item_mean_positive": float(pos["alpha_item"].mean()),
        "alpha_item_mean_negative": float(neg["alpha_item"].mean()),
        "alpha_user_ks_stat": float(ks_2samp(pos["alpha_user"], neg["alpha_user"]).statistic),
        "alpha_user_ks_pvalue": float(ks_2samp(pos["alpha_user"], neg["alpha_user"]).pvalue),
        "alpha_item_ks_stat": float(ks_2samp(pos["alpha_item"], neg["alpha_item"]).statistic),
        "alpha_item_ks_pvalue": float(ks_2samp(pos["alpha_item"], neg["alpha_item"]).pvalue),
        "spearman_positive": safe_spearman(pos["alpha_user"], pos["alpha_item"]),
        "spearman_negative": safe_spearman(neg["alpha_user"], neg["alpha_item"]),
    }
    return summary


def augment_fused_affinity(df):
    df = df.copy()
    mu_u = float(df["alpha_user"].mean())
    std_u = float(df["alpha_user"].std())
    mu_v = float(df["alpha_item"].mean())
    std_v = float(df["alpha_item"].std())
    if std_u < 1e-8:
        std_u = 1.0
    if std_v < 1e-8:
        std_v = 1.0
    df["alpha_user_z"] = (df["alpha_user"] - mu_u) / std_u
    df["alpha_item_z"] = (df["alpha_item"] - mu_v) / std_v
    df["alpha_fused"] = 0.5 * (df["alpha_user_z"] + df["alpha_item_z"])
    return df, {
        "mean_alpha_user": mu_u,
        "std_alpha_user": std_u,
        "mean_alpha_item": mu_v,
        "std_alpha_item": std_v,
    }


def compute_discriminability_stats(df, col):
    pos = df[df["label"] == "Positive"][col]
    neg = df[df["label"] == "Matched Negative"][col]
    ks = ks_2samp(pos, neg)
    return {
        "positive_median": float(pos.median()),
        "negative_median": float(neg.median()),
        "median_gap": float(pos.median() - neg.median()),
        "positive_mean": float(pos.mean()),
        "negative_mean": float(neg.mean()),
        "mean_gap": float(pos.mean() - neg.mean()),
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
    }


def make_plot(df, out_path):
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    sns.set_style(
        "whitegrid",
        {
            "axes.grid": True,
            "grid.color": "#EAEAEA",
            "grid.linestyle": "-",
            "axes.edgecolor": "#BBBBBB",
        },
    )

    pos_color = "#D55E00"
    neg_color = "#5B7C99"
    line_gray = "#888888"

    pos = df[df["label"] == "Positive"]
    neg = df[df["label"] == "Matched Negative"]
    rho_pos = spearmanr(pos["alpha_user"], pos["alpha_item"]).statistic
    rho_neg = spearmanr(neg["alpha_user"], neg["alpha_item"]).statistic

    pos_user_med = float(pos["alpha_user"].median())
    neg_user_med = float(neg["alpha_user"].median())
    pos_item_med = float(pos["alpha_item"].median())
    neg_item_med = float(neg["alpha_item"].median())

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))

    ax = axes[0]
    sns.ecdfplot(
        data=df,
        x="alpha_user",
        hue="label",
        stat="proportion",
        linewidth=2.0,
        palette={"Positive": pos_color, "Matched Negative": neg_color},
        ax=ax,
    )
    ax.set_title("(a) User-Centric Affinity")
    ax.set_xlabel(r"$\alpha^{(u)}$")
    ax.set_ylabel("ECDF")
    ax.text(
        0.03,
        0.10,
        rf"Median gap = {pos_user_med - neg_user_med:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.3,
    )
    ax.legend_.set_title(None)

    ax = axes[1]
    sns.ecdfplot(
        data=df,
        x="alpha_item",
        hue="label",
        stat="proportion",
        linewidth=2.0,
        palette={"Positive": pos_color, "Matched Negative": neg_color},
        ax=ax,
    )
    ax.set_title("(b) Item-Centric Affinity")
    ax.set_xlabel(r"$\alpha^{(v)}$")
    ax.set_ylabel("ECDF")
    ax.text(
        0.03,
        0.10,
        rf"Median gap = {pos_item_med - neg_item_med:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.3,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()

    ax = axes[2]
    scatter_n = min(3000, len(pos), len(neg))
    pos_scatter = pos.sample(n=scatter_n, random_state=2026)
    neg_scatter = neg.sample(n=scatter_n, random_state=2026)

    ax.scatter(
        neg_scatter["alpha_user"],
        neg_scatter["alpha_item"],
        s=7,
        alpha=0.10,
        color=neg_color,
        rasterized=True,
    )
    ax.scatter(
        pos_scatter["alpha_user"],
        pos_scatter["alpha_item"],
        s=7,
        alpha=0.10,
        color=pos_color,
        rasterized=True,
    )

    sns.kdeplot(
        data=neg,
        x="alpha_user",
        y="alpha_item",
        levels=5,
        color=neg_color,
        linewidths=1.2,
        ax=ax,
    )
    sns.kdeplot(
        data=pos,
        x="alpha_user",
        y="alpha_item",
        levels=5,
        color=pos_color,
        linewidths=1.2,
        ax=ax,
    )

    xmin = min(df["alpha_user"].min(), df["alpha_item"].min())
    xmax = max(df["alpha_user"].max(), df["alpha_item"].max())
    ax.plot([xmin, xmax], [xmin, xmax], ls="--", lw=1.0, color=line_gray, alpha=0.9)
    ax.text(
        0.03,
        0.97,
        rf"Spearman $\rho$: Pos={rho_pos:.2f}, Neg={rho_neg:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    ax.set_title("(c) Perspective-Affinity Plane")
    ax.set_xlabel(r"$\alpha^{(u)}$")
    ax.set_ylabel(r"$\alpha^{(v)}$")

    fig.subplots_adjust(wspace=0.28, left=0.07, right=0.99, top=0.92, bottom=0.19)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_fusion_ecdf_plot(df, out_path):
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    sns.set_style(
        "whitegrid",
        {
            "axes.grid": True,
            "grid.color": "#EAEAEA",
            "grid.linestyle": "-",
            "axes.edgecolor": "#BBBBBB",
        },
    )

    pos_color = "#D55E00"
    neg_color = "#5B7C99"
    panels = [
        ("alpha_user_z", r"(a) Standardized User-Centric Signal $\tilde{\alpha}^{(u)}$"),
        ("alpha_item_z", r"(b) Standardized Item-Centric Signal $\tilde{\alpha}^{(v)}$"),
        ("alpha_fused", r"(c) Fused Dual-Perspective Signal $\alpha^{(d)}$"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9))

    for idx, (ax, (col, title)) in enumerate(zip(axes, panels)):
        stats = compute_discriminability_stats(df, col)
        sns.ecdfplot(
            data=df,
            x=col,
            hue="label",
            stat="proportion",
            linewidth=2.0,
            palette={"Positive": pos_color, "Matched Negative": neg_color},
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Standardized Affinity Score")
        ax.set_ylabel("ECDF" if idx == 0 else "")
        ax.text(
            0.03,
            0.10,
            rf"Median gap = {stats['median_gap']:.3f}" + "\n" + rf"KS = {stats['ks_stat']:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.2,
        )
        if idx > 0 and ax.legend_ is not None:
            ax.legend_.remove()
        elif idx == 0 and ax.legend_ is not None:
            ax.legend_.set_title(None)

    fig.subplots_adjust(wspace=0.28, left=0.07, right=0.99, top=0.92, bottom=0.19)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    base = args.data_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    train_csr = ensure_binary_csr(load_pickle(os.path.join(base, "trnMat.pkl")))
    test_csr = ensure_binary_csr(load_pickle(os.path.join(base, "tstMat.pkl")))
    uu_csr = load_pickle(os.path.join(base, "uu10_csr")).tocsr().astype(np.float32)
    ii_csr = load_pickle(os.path.join(base, "ii10_csr")).tocsr().astype(np.float32)

    affinity_df, skipped = compute_affinity_dataframe(
        train_csr=train_csr,
        test_csr=test_csr,
        uu_csr=uu_csr,
        ii_csr=ii_csr,
        topk=args.topk,
        num_pos=args.num_pos,
        seed=args.seed,
    )

    dataset_tag = infer_dataset_tag(base)

    csv_path = os.path.join(out_dir, f"{dataset_tag}_perspective_affinity.csv")
    affinity_df.to_csv(csv_path, index=False)

    affinity_df, fusion_norm_stats = augment_fused_affinity(affinity_df)

    summary = compute_summary_stats(affinity_df)
    summary["skipped_due_to_negative_sampling"] = int(skipped)
    summary["topk"] = int(args.topk)
    summary["num_pos_requested"] = int(args.num_pos)
    summary["dataset_tag"] = dataset_tag
    summary["fusion_normalization"] = fusion_norm_stats
    summary["discriminability"] = {
        "alpha_user_z": compute_discriminability_stats(affinity_df, "alpha_user_z"),
        "alpha_item_z": compute_discriminability_stats(affinity_df, "alpha_item_z"),
        "alpha_fused": compute_discriminability_stats(affinity_df, "alpha_fused"),
    }

    summary_path = os.path.join(out_dir, f"{dataset_tag}_perspective_affinity_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    fig_path = os.path.join(out_dir, f"{dataset_tag}_perspective_dependence.pdf")
    make_plot(affinity_df, fig_path)

    fusion_csv_path = os.path.join(out_dir, f"{dataset_tag}_fusion_affinity.csv")
    affinity_df.to_csv(fusion_csv_path, index=False)

    fusion_fig_path = os.path.join(out_dir, f"{dataset_tag}_fusion_affinity_ecdf.pdf")
    make_fusion_ecdf_plot(affinity_df, fusion_fig_path)

    print("Saved:")
    print("  CSV    :", csv_path)
    print("  JSON   :", summary_path)
    print("  Figure :", fig_path)
    print("  Fusion CSV   :", fusion_csv_path)
    print("  Fusion Figure:", fusion_fig_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
