# evaluate.py  (patched)
# Key patches:
# - FIX: plot_delta_heatmap() uses pivot_table (supports multi-column "columns")
# - FIX: consistent delta direction via PREFERRED_LEVEL_ORDER + consistent naming
# - FIX: binarize_if_probabilistic() no longer converts NaNs to 0 for already-binary cols
# - FIX: avoid duplicate feat_thresholds definition
# - FIX: stronger/safer factor injection: ensure feature_set numeric ("17"/"25") exists even if not parseable from name
# - FIX: build_annotated_rationales() checks merged columns, not Series membership
# - FIX: safer string stripping on sentence columns (cast to str first)
# - FIX: avoid accidental column collisions when concatenating factors_df (drop existing)
# - NOTE: imports unchanged except minor typing convenience
import argparse
import sys

import os
import glob
from typing import List, Optional, Literal, Sequence, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score
from scipy.stats import wilcoxon
from itertools import combinations

from multi_prompt_configs import (
    EXTENDED_FEATURES,
    MASIS_FEATURES,
)

"""
nlprun -q jag -p standard -r 40G -c 2 -t 4:00:00 \
  -n eval-gpt-sheets \
  -o slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   mkdir -p slurm_logs && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/evaluate.py Datasets/FullTestFinal.xlsx"
"""

# Treat wh-qu as part of the MASIS/17 set (it may be derived from wh-qu1/wh-qu2)
MASIS17_FEATURES = list(MASIS_FEATURES)
if "wh-qu" not in MASIS17_FEATURES:
    MASIS17_FEATURES.append("wh-qu")

# enforce consistent delta direction (b - a)
PREFERRED_LEVEL_ORDER = {
    "ctx": ("noCTX", "CTX"),          # CTX - noCTX
    "instr": ("ZS", "FScot"),         # FScot - ZS
    "leg": ("noLeg", "Leg"),          # Leg - noLeg
    "feature_set": ("17", "25"),      # 25 - 17
    "trial": ("A", "B"),              # B - A
}

# one definition only
feat_thresholds = {
    "multiple-neg": 0.5,
}

output_dir = "Results"
os.makedirs(output_dir, exist_ok=True)


def drop_features_column(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["FEATURES", "Source", "idx"], errors="ignore")


def combine_wh_qu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines wh-qu1/wh-qu2 (or wh_qu1/wh_qu2) into a single wh-qu.
    If wh-qu already exists, do nothing.
    """
    out = df.copy()

    if "wh-qu" in out.columns:
        return out

    if "wh_qu" in out.columns and "wh-qu" not in out.columns:
        out = out.rename(columns={"wh_qu": "wh-qu"})

    a = "wh-qu1" if "wh-qu1" in out.columns else ("wh_qu1" if "wh_qu1" in out.columns else None)
    b = "wh-qu2" if "wh-qu2" in out.columns else ("wh_qu2" if "wh_qu2" in out.columns else None)

    if a and b:
        out["wh-qu"] = out[[a, b]].max(axis=1)
        out = out.drop(columns=[a, b])

    return out

SENTENCE_ALIASES = ["sentence", "Sentence", "SENTENCE", "text", "Text", "utterance", "Utterance", "UTTERANCE"]

def normalize_sentence_column(df: pd.DataFrame, sheet_name: str) -> Optional[pd.DataFrame]:
    """
    Ensure df has a column named 'sentence'.
    - If it has an alias (Sentence/text/utterance), rename it.
    - If it has none, return None (caller should skip sheet).
    """
    cols = list(df.columns)
    if "sentence" in cols:
        return df

    for c in SENTENCE_ALIASES:
        if c in cols:
            out = df.rename(columns={c: "sentence"}).copy()
            return out

    # last resort: case-insensitive match
    lower_map = {str(c).strip().lower(): c for c in cols}
    if "sentence" in lower_map:
        out = df.rename(columns={lower_map["sentence"]: "sentence"}).copy()
        return out

    print(f"[WARN] Sheet '{sheet_name}' has no 'sentence' column (cols={cols[:15]}...). Skipping.")
    return None


def parse_factors(sheet_name: str) -> dict:
    """
    Parse condition factors out of sheet names like:
      Old style: GPT_17_ZS_CTX_A / GEMINI_25_FSCOT_noCTX_B
      New style: PHI_ZS_noCTX_noLeg / GEMINI_FScot_CTX2t_Leg / PHI4_ZScot_CTX1t_noLeg
    Returns dict with keys:
      feature_set in {"17","25"} or None
      instr in {"ZS","FS","ZScot","FScot"} or None
      ctx in {"noCTX","CTX1t","CTX2t"} or None
      leg in {"Leg","noLeg"} or None
      trial in {"A","B"} or None
    """
    m = str(sheet_name).upper()
    toks = [t for t in m.split("_") if t]

    feature_set = None
    if "17" in toks or "_17_" in f"_{m}_":
        feature_set = "17"
    elif "25" in toks or "_25_" in f"_{m}_":
        feature_set = "25"

    instr = None
    if "FSCOT" in toks:
        instr = "FScot"
    elif "ZSCOT" in toks:
        instr = "ZScot"
    elif "FS" in toks:
        instr = "FS"
    elif "ZS" in toks:
        instr = "ZS"

    ctx = None
    if "NOCTX" in toks:
        ctx = "noCTX"
    elif "CTX2T" in toks:
        ctx = "CTX2t"
    elif "CTX1T" in toks:
        ctx = "CTX1t"
    elif "CTX" in toks:
        ctx = "CTX"

    leg = None
    if "NOLEG" in toks:
        leg = "noLeg"
    elif "LEG" in toks:
        leg = "Leg"

    trial = None
    if toks and toks[-1] in ("A", "B"):
        trial = toks[-1]

    return {"feature_set": feature_set, "instr": instr, "ctx": ctx, "leg": leg, "trial": trial}

def safe_strip_sentence_col(df: pd.DataFrame) -> pd.DataFrame:
    if "sentence" in df.columns:
        df = df.copy()
        df["sentence"] = df["sentence"].astype(str).str.strip()
    return df


def build_annotated_rationales(
    pred_df: pd.DataFrame,
    rationale_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    features: list[str],
    only_disagreements: bool = True,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    pred_df = safe_strip_sentence_col(pred_df.copy())
    rationale_df = safe_strip_sentence_col(rationale_df.copy())
    truth_df = safe_strip_sentence_col(truth_df.copy())

    merged = truth_df.merge(pred_df, on="sentence", suffixes=("_gold", "_pred"))
    merged = merged.merge(rationale_df, on="sentence", how="left", suffixes=("", "_rat"))

    rows = []
    merged_cols = set(merged.columns)

    for _, row in merged.iterrows():
        sent = row["sentence"]
        for feat in features:
            gold_col = f"{feat}_gold"
            pred_col = f"{feat}_pred"
            rat_col = feat

            if gold_col not in merged_cols or pred_col not in merged_cols:
                continue

            gold_v = row.get(gold_col, None)
            pred_v = row.get(pred_col, None)

            if pd.isna(gold_v) or pd.isna(pred_v):
                continue

            try:
                gold_v = int(gold_v)
                pred_v = int(pred_v)
            except Exception:
                continue

            rationale = row.get(rat_col, "")
            if pd.isna(rationale):
                rationale = ""

            if only_disagreements and gold_v == pred_v:
                continue

            rows.append(
                {
                    "sentence": sent,
                    "feature": feat,
                    "gold": gold_v,
                    "pred": pred_v,
                    "model_rationale": rationale,
                }
            )

    out_df = pd.DataFrame(rows)
    if max_rows is not None:
        out_df = out_df.head(max_rows)
    return out_df


def plot_delta_heatmap(
    delta_df: pd.DataFrame,
    value_col: str,
    title: str,
    save_path: str,
    figsize=(14, 10),
):
    """
    delta_df must contain:
      - 'feature'
      - value_col
      - optionally additional factor columns (e.g., feature_set/instr/trial)
    Heatmap columns will be those factor columns (possibly multi-index if >1).
    """
    if delta_df.empty:
        print(f"[plot_delta_heatmap] empty: {title}")
        return

    if "feature" not in delta_df.columns or value_col not in delta_df.columns:
        print(f"[plot_delta_heatmap] missing columns for {title}")
        return

    col_cols = [c for c in delta_df.columns if c not in ("feature", value_col) and not c.startswith("d_")]

    tmp = delta_df.copy()
    if not col_cols:
        tmp["_all"] = ""
        col_cols = ["_all"]

    # pivot_table supports list of columns -> multiindex columns
    pivot = tmp.pivot_table(
        index="feature",
        columns=col_cols,
        values=value_col,
        aggfunc="mean",
    )

    plt.figure(figsize=figsize)
    sns.heatmap(pivot, annot=False, cmap="RdBu_r", center=0.0)
    plt.title(title)
    plt.ylabel("feature")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_overall_micro_f1_scores(
    eval_dfs: List[pd.DataFrame],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
    align: Literal["union", "intersection"] = "intersection",
):
    if align == "intersection":
        shared = None
        for df in eval_dfs:
            feats = set(df["feature"].unique())
            shared = feats if shared is None else shared.intersection(feats)
        shared = shared or set()
        eval_dfs = [df[df["feature"].isin(shared)].copy() for df in eval_dfs]
        print(f"[micro-F1] Using {len(shared)} shared features: {sorted(shared)}")

    micro_scores = {}
    for df in eval_dfs:
        if df.empty:
            continue
        model = df["model"].iloc[0]
        TP = int(df["TP"].sum())
        FP = int(df["FP"].sum())
        FN = int(df["FN"].sum())
        denom = (2 * TP + FP + FN)
        micro_f1 = (2 * TP / denom) if denom > 0 else 0.0
        micro_scores[model] = micro_f1

    out = pd.DataFrame.from_dict(micro_scores, orient="index", columns=["micro_F1"]).sort_values("micro_F1", ascending=False)

    ax = out.plot(kind="bar", figsize=figsize, legend=False)
    ax.set_ylabel("micro-F1")
    ax.set_title(
        "Overall micro-F1 by model (shared features only)"
        if align == "intersection"
        else "Overall micro-F1 by model (each model’s evaluated features; not strictly comparable)"
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_overall_f1_scores(
    eval_dfs: List[pd.DataFrame],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
    align: Literal["union", "intersection"] = "intersection",
):
    overall_f1_scores = {}

    if align == "intersection":
        all_features = set()
        for df in eval_dfs:
            if not df.empty:
                all_features.update(df["feature"].unique())

        shared_features = all_features.copy()
        for df in eval_dfs:
            if not df.empty:
                shared_features = shared_features.intersection(set(df["feature"].unique()))

        eval_dfs = [(df[df["feature"].isin(shared_features)].copy() if not df.empty else df) for df in eval_dfs]
        print(f"Using {len(shared_features)} shared features for overall F1: {sorted(shared_features)}")

    for df in eval_dfs:
        if not df.empty:
            model_name = df["model"].iloc[0]
            overall_f1_scores[model_name] = df["f1"].mean()

    overall_f1_df = (
        pd.DataFrame.from_dict(overall_f1_scores, orient="index", columns=["F1 Score"])
        .sort_values(by="F1 Score", ascending=False)
    )

    ax = overall_f1_df.plot(kind="bar", figsize=figsize, legend=False)
    ax.set_ylabel("F1 Score")
    ax.set_title(
        "Overall F1 Scores by Model (shared features only)"
        if align == "intersection"
        else "Overall F1 Scores by Model (each model’s evaluated features; not strictly comparable)"
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_f1_scores_per_feature(
    eval_dfs: List[pd.DataFrame],
    save_path: Optional[str] = None,
    figsize: tuple = (14, 10),
    align: Literal["union", "intersection"] = "intersection",
):
    all_eval = pd.concat(eval_dfs, ignore_index=True)
    pivot = all_eval.pivot(index="feature", columns="model", values="f1")

    if align == "intersection":
        present_counts = pivot.notna().sum(axis=1)
        pivot = pivot[present_counts == pivot.shape[1]]
        print(f"Showing {len(pivot)} shared features in heatmap")

    pivot = pivot.fillna(0.0).sort_index()

    plt.figure(figsize=figsize)
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.0, vmax=1.0, cbar_kws={"label": "F1 score"})
    title = "F1 Scores per Feature by Model (Shared Features Only)" if align == "intersection" else "F1 Scores per Feature by Model"
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel("Feature")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_per_feature_confusion_matrix(cm_data: pd.DataFrame, model_name: str, features: list[str], save_path: Optional[str] = None):

    if cm_data.empty:
        print(f"[INFO] No confusion-matrix data for {model_name}; skipping heatmap.")
        return

    plt.figure(figsize=(14, 10))
    sns.heatmap(cm_data, annot=True, fmt="d", cmap="Blues", xticklabels=["TP", "FP", "FN", "TN"], yticklabels=features)
    plt.title(f"Per-Feature TP/FP/FN/TN for {model_name}")
    plt.xlabel("Count type")
    plt.ylabel("Feature")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_model(model_df: pd.DataFrame, truth_df: pd.DataFrame, model_name: str, features: list[str], output_base: str) -> pd.DataFrame:
    print(f"\n=== {model_name} Evaluation ===")

    if model_df is None:
        print(f"{model_name} data not available. Skipping evaluation.")
        return pd.DataFrame()

    model_df = safe_strip_sentence_col(model_df.drop_duplicates(subset="sentence"))
    truth_df = safe_strip_sentence_col(truth_df.drop_duplicates(subset="sentence"))

    model_df = combine_wh_qu(model_df)
    truth_df = combine_wh_qu(truth_df)

    available_features = [feat for feat in features if feat in model_df.columns and feat in truth_df.columns]
    shared_columns = ["sentence"] + available_features

    model_df = model_df[shared_columns]
    truth_df = truth_df[shared_columns]

    summary = {k: [] for k in ["model", "feature", "accuracy", "precision", "recall", "f1", "TP", "FP", "FN", "TN"]}
    skipped_features = []
    cm_data_rows = []

    for feat in available_features:
        merged = pd.merge(
            truth_df[["sentence", feat]],
            model_df[["sentence", feat]],
            on="sentence",
            how="inner",
        ).rename(columns={f"{feat}_x": "y_true", f"{feat}_y": "y_pred"})

        merged = merged.dropna(subset=["y_true", "y_pred"])
        if merged.empty:
            print(f"Skipping {feat} — no data after dropping NaNs.")
            skipped_features.append(feat)
            continue

        # cast after dropna
        y_true = merged["y_true"].astype(int)
        y_pred = merged["y_pred"].astype(int)

        if y_true.sum() == 0 and y_pred.sum() == 0:
            print(f"Skipping {feat} — no positive instances in ground truth and predictions.")
            skipped_features.append(feat)
            continue

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1v = f1_score(y_true, y_pred, zero_division=0)

        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        TP = int(cm[0, 0])
        FN = int(cm[0, 1])
        FP = int(cm[1, 0])
        TN = int(cm[1, 1])

        summary["model"].append(model_name)
        summary["feature"].append(feat)
        summary["accuracy"].append(acc)
        summary["precision"].append(prec)
        summary["recall"].append(rec)
        summary["f1"].append(f1v)
        summary["TP"].append(TP)
        summary["FP"].append(FP)
        summary["FN"].append(FN)
        summary["TN"].append(TN)

        cm_data_rows.append([TP, FP, FN, TN])

    filtered_features = [feat for feat in available_features if feat not in skipped_features]
    cm_df = pd.DataFrame(cm_data_rows, columns=["TP", "FP", "FN", "TN"], index=filtered_features)

    if cm_df.empty:
        print(f"[INFO] No non-empty features for {model_name}; skipping.")
        return pd.DataFrame()

    results = pd.DataFrame(summary)
    print("\n=== Summary Metrics ===")
    print(results.round(4))

    if not results.empty:
        print("\n=== Macro Averages ===")
        print(results[["accuracy", "precision", "recall", "f1"]].mean().round(4))
        print("\n=== Per-feature TP/FP/FN/TN ===")
        print(results[["feature", "TP", "FP", "FN", "TN"]].to_string(index=False))

    print(f"\n=== Skipped {len(skipped_features)} feature(s): {skipped_features} ===")
    return results


def plot_model_metrics(
    *,
    eval_dfs: Optional[List[pd.DataFrame]] = None,
    err_counts: Optional[pd.DataFrame] = None,
    metric: Literal["f1", "errors"] = "f1",
    style: Literal["bar", "heatmap"] = "bar",
    title: Optional[str] = None,
    figsize: tuple = (12, 6),
    rotation: int = 45,
    save_path: Optional[str] = None,
    align: Literal["union", "intersection"] = "union",
    annotate_heatmap: bool = True,
) -> None:
    def _prepare_f1_pivot(dfs: List[pd.DataFrame]) -> pd.DataFrame:
        all_eval = pd.concat(dfs, ignore_index=True)
        pivot = all_eval.pivot(index="feature", columns="model", values="f1")

        if align == "intersection":
            present_counts = pivot.notna().sum(axis=1)
            pivot = pivot[present_counts == pivot.shape[1]]

        return pivot.fillna(0.0).sort_index()

    def _prepare_err_pivot(df: pd.DataFrame) -> pd.DataFrame:
        if {"feature", "model", "errors"}.issubset(df.columns):
            p = df.pivot(index="feature", columns="model", values="errors")
        else:
            p = df.copy()
        return p.fillna(0).sort_index()

    if metric == "f1":
        if not eval_dfs:
            raise ValueError("metric='f1' requires eval_dfs=[...]")
        pivot = _prepare_f1_pivot(eval_dfs)
        default_title = "Model F1 by AAE feature"
        vmin, vmax = 0.0, 1.0
        cbar_label = "F1 score"
        ylab, xlab = "AAE feature", "Model"
        fmt, cmap = ".2f", "RdYlGn"
    else:
        if err_counts is None:
            raise ValueError("metric='errors' requires err_counts=DataFrame")
        pivot = _prepare_err_pivot(err_counts)
        default_title = "Errors per feature by model"
        vmin, vmax = None, None
        cbar_label = "Number of errors"
        ylab, xlab = "Feature", "Model"
        fmt, cmap = ".0f", "Reds"

    title = title or default_title

    if style == "bar":
        ax = pivot.plot(kind="bar", figsize=figsize)
        ax.set_ylabel(cbar_label)
        ax.set_title(title)
        ax.legend(title="Model")
        plt.xticks(rotation=rotation, ha="right")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    elif style == "heatmap":
        plt.figure(figsize=figsize)
        sns.heatmap(
            pivot,
            annot=annotate_heatmap,
            fmt=fmt,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar_kws={"label": cbar_label},
        )
        plt.title(title)
        plt.ylabel(ylab)
        plt.xlabel(xlab)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        raise ValueError("style must be 'bar' or 'heatmap'")

# ==========================================
# 1. Parsing Helper (Updates your Metadata)
# ==========================================
# ==========================================
# [START] NEW VISUALIZATION FUNCTIONS
# ==========================================

def parse_model_config(name: str, model_name_col: str = None):
    """
    Parses sheet/model names to extract Model and Config string.

    Handles new naming convention:
      PHI_ZS_noCTX_noLeg, GEMINI_FScot_CTX2t_Leg, PHI4_ZScot_CTX1t_noLeg
    And old convention:
      GPT_17_ZS_CTX_A, GEMINI_25_FSCOT_noCTX_B

    If model_name_col is provided (from combine_predictions.py model_name column),
    use it to identify the model.
    """
    base = str(name).upper()

    # --- 1. Identify Model ---
    if model_name_col:
        model = str(model_name_col)
    elif "MODERN-BERT" in base or "MODERNBERT" in base: model = "Modern-BERT"
    elif "PHI-4" in base or "PHI4" in base: model = "Phi-4"
    elif "PHI" in base and "PHI" == base.split("_")[0]: model = "Phi-4"
    elif "BERT" in base: model = "BERT"
    elif "GPT" in base: model = "GPT"
    elif "GEMINI" in base or "GEM" == base.split("_")[0]: model = "Gemini"
    elif "QWEN25" in base or "QWEN" == base.split("_")[0]: model = "Qwen2.5"
    else: model = base.split("_")[0]

    # --- 2. Identify Configuration components ---
    # Instruction Strategy
    if "FSCOT" in base: instr = "Few-Shot CoT"
    elif "ZSCOT" in base: instr = "Zero-Shot CoT"
    elif "_FS_" in f"_{base}_" or base.endswith("_FS"): instr = "Few-Shot"
    elif "ICL" in base: instr = "Few-Shot (ICL)"
    elif "ZS" in base: instr = "Zero-Shot"
    else: instr = "Standard"

    # Context
    if "NOCTX" in base or "NO_CTX" in base:
        ctx = "No Context"
    elif "CTX2T" in base:
        ctx = "Context (2-turn)"
    elif "CTX1T" in base:
        ctx = "Context (1-turn)"
    elif "CTX" in base:
        ctx = "Context"
    else:
        ctx = ""

    # --- 3. Dialect Legitimacy ---
    if "NOLEG" in base:
        legit = "No Legitimacy"
    elif "LEG" in base:
        legit = "Legitimacy"
    else:
        legit = ""

    # Combine into one descriptive config label
    parts = [p for p in [instr, ctx, legit] if p]
    config = " + ".join(parts)

    return model, config


def plot_faceted_model_comparison(
    eval_dfs: list[pd.DataFrame], 
    save_path: str = "faceted_model_comparison.png"
):
    """
    Plots a grid of heatmaps for all configurations.
    """
    if not eval_dfs: return
    master_df = pd.concat(eval_dfs, ignore_index=True)

    if 'config' not in master_df.columns or 'clean_model' not in master_df.columns:
        print("[WARN] Missing columns for plotting. Skipping.")
        return

    # Pivot Data
    try:
        pivot = master_df.pivot_table(
            index='feature', 
            columns=['config', 'clean_model'], 
            values='f1', 
            aggfunc='mean'
        )
    except KeyError: return

    configs = sorted(master_df['config'].unique())
    n_configs = len(configs)
    
    # --- Better Layout Logic for Many Facets ---
    # If > 5 configs, wrap into multiple rows
    import math
    ncols = min(n_configs, 4) 
    nrows = math.ceil(n_configs / ncols)
    
    fig_width = 6 * ncols
    fig_height = max(8, len(pivot) * 0.5) * nrows # Scale height by rows
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), sharey=True, squeeze=False)
    axes = axes.flatten() # Flatten 2D array to 1D for easy iteration

    for i, cfg in enumerate(configs):
        ax = axes[i]
        if cfg in pivot.columns.get_level_values(0):
            data = pivot[cfg]
            data = data.reindex(sorted(data.columns), axis=1)
            
            sns.heatmap(
                data, 
                ax=ax, 
                cmap="RdYlGn", 
                vmin=0.0, vmax=1.0, 
                annot=True, fmt=".2f", 
                cbar=(i == ncols - 1) # Only on rightmost plot of first row (approx)
            )
            ax.set_title(cfg, fontsize=10, fontweight='bold', pad=10)
            ax.set_xlabel("")
            if i % ncols != 0: ax.set_ylabel("") # Only Y-label on left-most cols
        else:
            ax.axis('off')

    # Turn off unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Model Performance by Configuration (Instr + Ctx + Legit + Task)", fontsize=16, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved faceted heatmap to {save_path}")
    plt.close()






def plot_aggregated_model_performance(
    eval_dfs: list[pd.DataFrame], 
    save_path: str = "aggregated_model_performance.png"
):
    if not eval_dfs: return
    master_df = pd.concat(eval_dfs, ignore_index=True)
    
    if 'config' not in master_df.columns or 'clean_model' not in master_df.columns:
        return

    summary = master_df.groupby(['clean_model', 'config'])['f1'].mean().reset_index()
    
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=summary,
        x='clean_model',
        y='f1',
        hue='config',
        palette="viridis",
        edgecolor="black"
    )
    plt.title("Average F1 Score: Model vs. Configuration", fontsize=15)
    plt.ylim(0, 1.05)
    plt.xlabel("Model")
    plt.ylabel("Mean F1 Score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved aggregated bar chart to {save_path}")
    plt.close()


# ==================== SUMMARY VISUALIZATIONS ====================


def plot_config_leaderboard(
    all_eval: pd.DataFrame,
    save_path: str,
    figsize: tuple = (10, 8),
) -> None:
    """Horizontal bar chart of configs ranked by macro-F1."""
    macro = (
        all_eval.groupby("model")["f1"]
        .mean()
        .sort_values(ascending=True)
        .reset_index(name="macro_f1")
    )

    factor_cols = [c for c in ("instr", "ctx", "leg") if c in all_eval.columns]
    if factor_cols:
        factor_labels = all_eval.groupby("model")[factor_cols].first().reset_index()
        macro = macro.merge(factor_labels, on="model", how="left")

    fig, ax = plt.subplots(figsize=figsize)

    # Color by instruction type if available
    if "instr" in macro.columns:
        unique_instr = sorted(macro["instr"].dropna().unique())
        cmap = plt.cm.get_cmap("Set2", max(len(unique_instr), 1))
        palette = {lvl: cmap(i) for i, lvl in enumerate(unique_instr)}
        colors = [palette.get(row.get("instr"), "steelblue") for _, row in macro.iterrows()]
    else:
        palette = None
        colors = "steelblue"

    ax.barh(macro["model"], macro["macro_f1"], color=colors)

    # Annotate bars with factor levels
    if factor_cols:
        for i, (_, row) in enumerate(macro.iterrows()):
            parts = [str(row[c]) for c in factor_cols if pd.notna(row.get(c))]
            if parts:
                ax.text(
                    row["macro_f1"] + 0.005, i,
                    " | ".join(parts),
                    va="center", fontsize=7, color="gray",
                )

    ax.set_xlabel("Macro F1")
    ax.set_title("Configuration Leaderboard (ranked by macro-F1)")
    ax.set_xlim(0, min(1.05, macro["macro_f1"].max() + 0.1))

    if palette:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=palette[k], label=k) for k in sorted(palette)]
        ax.legend(handles=legend_elements, title="Instruction", loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved config leaderboard to {save_path}")


def plot_variable_effects(
    summary_df: Optional[pd.DataFrame],
    save_path: str,
    figsize: tuple = (8, 6),
) -> None:
    """Forest plot: mean delta ± 95% CI per factor comparison."""
    if summary_df is None or summary_df.empty:
        print("[INFO] No significance summary to plot; skipping forest plot.")
        return

    df = summary_df.copy()
    df["label"] = df.apply(
        lambda r: f"{r['factor']}: {r['comparison'].replace('_minus_', ' vs ')}", axis=1
    )
    df = df.sort_values("mean_delta", key=abs, ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)

    for i, row in df.iterrows():
        color = "firebrick" if row.get("significant") else "gray"
        ci_lo = row.get("ci_lo", row["mean_delta"])
        ci_hi = row.get("ci_hi", row["mean_delta"])

        ax.plot([ci_lo, ci_hi], [i, i], color=color, linewidth=2, solid_capstyle="round")
        ax.plot(row["mean_delta"], i, "o", color=color, markersize=7, zorder=5)

    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["label"].tolist(), fontsize=9)
    ax.set_xlabel("Mean ΔF1 (b − a)")
    ax.set_title("Variable Effects on F1 (forest plot)")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="firebrick", linestyle="-", label="Significant (p < 0.05)"),
        Line2D([0], [0], marker="o", color="gray", linestyle="-", label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved variable effects forest plot to {save_path}")


def generate_summary_table(
    all_eval: pd.DataFrame,
    save_path: str,
) -> pd.DataFrame:
    """One-stop leaderboard CSV: rank, model, macro_f1, micro_f1, factor levels."""
    macro = all_eval.groupby("model")["f1"].mean().reset_index(name="macro_f1")

    micro_rows = []
    for model_name, grp in all_eval.groupby("model"):
        tp = grp["TP"].sum()
        fp = grp["FP"].sum()
        fn = grp["FN"].sum()
        denom = 2 * tp + fp + fn
        micro_f1 = (2 * tp / denom) if denom > 0 else 0.0
        micro_rows.append({"model": model_name, "micro_f1": round(micro_f1, 4)})
    micro = pd.DataFrame(micro_rows)

    summary = macro.merge(micro, on="model", how="left")

    factor_cols = [c for c in ("instr", "ctx", "leg", "feature_set", "trial") if c in all_eval.columns]
    if factor_cols:
        factors = all_eval.groupby("model")[factor_cols].first().reset_index()
        summary = summary.merge(factors, on="model", how="left")

    summary = summary.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    summary.insert(0, "rank", summary.index + 1)
    summary["macro_f1"] = summary["macro_f1"].round(4)

    summary.to_csv(save_path, index=False)
    print(f"[INFO] Wrote summary leaderboard to {save_path} ({len(summary)} configs)")
    return summary


def detect_feature_set(sheet_name: str, df: pd.DataFrame) -> Optional[str]:
    """
    Returns: "masis" or "extended" or None.
    """
    sheet_up = str(sheet_name).upper()
    cols = set(df.columns)

    if "_17_" in sheet_up or sheet_up.endswith("_17") or sheet_up.startswith("17_") or "_17" in sheet_up:
        return "masis"
    if "_25_" in sheet_up or sheet_up.endswith("_25") or sheet_up.startswith("25_") or "_25" in sheet_up:
        return "extended"

    masis_overlap = len(cols.intersection(MASIS_FEATURES))
    extended_overlap = len(cols.intersection(EXTENDED_FEATURES))

    if extended_overlap > masis_overlap:
        return "extended"
    if masis_overlap > 0:
        return "masis"
    return None


def binarize_if_probabilistic(df: pd.DataFrame, features: list[str], model_name: str) -> pd.DataFrame:
    """
    Convert probabilistic columns to {0,1} using thresholds.
    IMPORTANT PATCH: do not fill NaNs with 0 for already-binary columns; preserve NaNs.
    """
    out = df.copy()
    for feat in features:
        if feat not in out.columns:
            continue

        s = out[feat]

        uniq = set(pd.Series(s.dropna()).unique().tolist())
        if uniq.issubset({0, 1, 0.0, 1.0, True, False}):
            # preserve NaNs; cast safely later after dropna in evaluation
            out[feat] = pd.to_numeric(s, errors="coerce")
            continue

        vals = pd.to_numeric(s, errors="coerce")
        if vals.notna().sum() == 0:
            continue  # don't clobber

        thr = feat_thresholds.get(feat, 0.5)
        out[feat] = (vals >= thr).astype(int)

    return out


METRICS_DEFAULT = ("precision", "recall", "f1")


def safe_pivot_get(pivot: pd.DataFrame, metric: str, level: str) -> pd.Series:
    key = (metric, level)
    if key in pivot.columns:
        return pivot[key]
    return pd.Series(index=pivot.index, data=pd.NA, dtype="float64")


def compute_pairwise_deltas(
    df: pd.DataFrame,
    *,
    index_cols: Sequence[str],
    factor_col: str,
    a_level: str,
    b_level: str,
    metrics: Sequence[str] = METRICS_DEFAULT,
    require_metric: str = "f1",
    aggfunc: str = "mean",
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute (b_level - a_level) for each metric, only where both levels exist.
    """
    pivot = df.pivot_table(
        index=list(index_cols),
        columns=factor_col,
        values=list(metrics),
        aggfunc=aggfunc,
    )

    have_both = safe_pivot_get(pivot, require_metric, a_level).notna() & safe_pivot_get(pivot, require_metric, b_level).notna()
    pivot = pivot[have_both]

    out = {}
    for m in metrics:
        d = safe_pivot_get(pivot, m, b_level) - safe_pivot_get(pivot, m, a_level)
        colname = f"d_{m}_{b_level}_minus_{a_level}"
        if prefix:
            colname = f"{prefix}{colname}"
        out[colname] = d

    return pd.DataFrame(out).reset_index()


# ───────────────────────────────────────────────────────────────
# Statistical Significance Tests
# ───────────────────────────────────────────────────────────────

def compute_factor_significance(
    df: pd.DataFrame,
    *,
    factor_col: str,
    a_level: str,
    b_level: str,
    metric: str = "f1",
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    For each feature, compute mean *metric* at a_level and b_level
    (averaging over all other experimental conditions), then run:
      1. Paired Wilcoxon signed-rank test on the per-feature differences
      2. Bootstrap 95 % CI on the mean difference
      3. Cohen's d effect size
    Returns a DataFrame with one row per feature plus an ``_OVERALL`` row.
    """
    group_a = df[df[factor_col] == a_level].groupby("feature")[metric].mean()
    group_b = df[df[factor_col] == b_level].groupby("feature")[metric].mean()

    shared = group_a.index.intersection(group_b.index)
    if shared.empty:
        return pd.DataFrame()

    vals_a = group_a.loc[shared].values
    vals_b = group_b.loc[shared].values
    deltas = vals_b - vals_a

    # Per-feature rows
    rows = []
    for feat, va, vb, d in zip(shared, vals_a, vals_b, deltas):
        rows.append({
            "feature": feat,
            f"{metric}_{a_level}": round(va, 4),
            f"{metric}_{b_level}": round(vb, 4),
            "delta": round(d, 4),
        })

    # Wilcoxon signed-rank (needs ≥6 non-zero differences)
    nonzero = deltas[deltas != 0]
    if len(nonzero) >= 6:
        stat, p_val = wilcoxon(deltas, alternative="two-sided")
    else:
        stat, p_val = float("nan"), float("nan")

    # Bootstrap CI
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(deltas, size=len(deltas), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    ci_lo = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    # Cohen's d
    mean_d = float(deltas.mean())
    std_d = float(deltas.std(ddof=1))
    cohens_d = mean_d / std_d if std_d > 0 else float("nan")

    overall = {
        "feature": "_OVERALL",
        f"{metric}_{a_level}": round(float(vals_a.mean()), 4),
        f"{metric}_{b_level}": round(float(vals_b.mean()), 4),
        "delta": round(mean_d, 4),
        "wilcoxon_stat": round(float(stat), 4) if not np.isnan(stat) else None,
        "wilcoxon_p": round(float(p_val), 6) if not np.isnan(p_val) else None,
        "bootstrap_ci_lo": round(ci_lo, 4),
        "bootstrap_ci_hi": round(ci_hi, 4),
        "cohens_d": round(cohens_d, 4) if not np.isnan(cohens_d) else None,
        "n_features": len(deltas),
        "significant": bool(p_val < alpha) if not np.isnan(p_val) else None,
    }

    # Copy global stats onto each per-feature row for CSV convenience
    for row in rows:
        for k in ("wilcoxon_stat", "wilcoxon_p", "bootstrap_ci_lo",
                   "bootstrap_ci_hi", "cohens_d", "n_features", "significant"):
            row[k] = overall[k]

    rows.append(overall)
    return pd.DataFrame(rows)


def _holm_bonferroni(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Holm–Bonferroni step-down correction for multiple comparisons."""
    n = len(p_values)
    order = np.argsort(p_values)
    corrected = np.full(n, np.nan)
    for rank, idx in enumerate(order):
        if np.isnan(p_values[idx]):
            corrected[idx] = np.nan
        else:
            corrected[idx] = min(float(p_values[idx]) * (n - rank), 1.0)
    # Enforce monotonicity
    sorted_corr = corrected[order]
    for i in range(1, n):
        if not np.isnan(sorted_corr[i]):
            sorted_corr[i] = max(sorted_corr[i], sorted_corr[i - 1])
    corrected[order] = sorted_corr
    return corrected


def compute_multilevel_significance(
    df: pd.DataFrame,
    *,
    factor_col: str,
    levels: Optional[List[str]] = None,
    metric: str = "f1",
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    correction: str = "holm",
    seed: int = 42,
) -> pd.DataFrame:
    """
    All pairwise significance tests between multiple levels of a factor.
    Applies Holm–Bonferroni correction across the family of comparisons.
    """
    if levels is None:
        levels = sorted(df[factor_col].dropna().unique().tolist())

    pairs = list(combinations(levels, 2))
    summaries: list[dict] = []
    raw_ps: list[float] = []

    for a_lev, b_lev in pairs:
        sub = df[df[factor_col].isin([a_lev, b_lev])].copy()
        result = compute_factor_significance(
            sub,
            factor_col=factor_col,
            a_level=a_lev,
            b_level=b_lev,
            metric=metric,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            seed=seed,
        )
        if result.empty:
            continue
        row = result[result["feature"] == "_OVERALL"].iloc[0].to_dict()
        row["comparison"] = f"{b_lev}_minus_{a_lev}"
        row["a_level"] = a_lev
        row["b_level"] = b_lev
        summaries.append(row)
        raw_ps.append(row.get("wilcoxon_p") if row.get("wilcoxon_p") is not None else np.nan)

    if not summaries:
        return pd.DataFrame()

    out = pd.DataFrame(summaries)
    p_arr = np.array([p if p is not None else np.nan for p in raw_ps], dtype=float)
    out["p_corrected"] = _holm_bonferroni(p_arr, alpha)
    out["significant_corrected"] = out["p_corrected"] < alpha
    return out


def auto_significance_tests(
    df: pd.DataFrame,
    *,
    factors: List[str],
    output_base: str,
    prefix: str,
    metric: str = "f1",
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
) -> Optional[pd.DataFrame]:
    """
    Run significance tests for all factors, mirroring auto_pairwise_deltas().
    Two-level factors  -> Wilcoxon + bootstrap CI + Cohen's d.
    Multi-level factors -> all pairwise comparisons with Holm correction.
    Returns the combined summary DataFrame (one row per factor comparison).
    """
    work = df.copy().dropna(subset=["feature"])
    factors = [f for f in factors if f in work.columns]

    all_summaries: list[dict] = []

    for fac in factors:
        levels = sorted(work[fac].dropna().unique().tolist())

        if len(levels) < 2:
            continue

        sub = work.dropna(subset=[fac]).copy()

        if len(levels) == 2:
            # Respect PREFERRED_LEVEL_ORDER for direction
            if fac in PREFERRED_LEVEL_ORDER:
                pref_a, pref_b = PREFERRED_LEVEL_ORDER[fac]
                if set(levels) == {pref_a, pref_b}:
                    a_level, b_level = pref_a, pref_b
                else:
                    a_level, b_level = levels[0], levels[1]
            else:
                a_level, b_level = levels[0], levels[1]

            result = compute_factor_significance(
                sub, factor_col=fac, a_level=a_level, b_level=b_level,
                metric=metric, n_bootstrap=n_bootstrap, alpha=alpha,
            )
            if result.empty:
                continue

            csv_path = os.path.join(
                output_base,
                f"{prefix}significance_{fac}__{b_level}_minus_{a_level}.csv",
            )
            result.to_csv(csv_path, index=False)

            overall = result[result["feature"] == "_OVERALL"].iloc[0]
            print(f"\n[SIG] {fac}: {b_level} vs {a_level}")
            print(f"  Mean delta  = {overall['delta']:.4f}")
            print(f"  Wilcoxon p  = {overall['wilcoxon_p']}")
            print(f"  95% CI      = [{overall['bootstrap_ci_lo']:.4f}, {overall['bootstrap_ci_hi']:.4f}]")
            print(f"  Cohen's d   = {overall['cohens_d']}")
            print(f"  Significant = {overall['significant']}")
            print(f"  Wrote {csv_path}")

            all_summaries.append({
                "factor": fac,
                "comparison": f"{b_level}_minus_{a_level}",
                "mean_delta": overall["delta"],
                "wilcoxon_p": overall["wilcoxon_p"],
                "ci_lo": overall["bootstrap_ci_lo"],
                "ci_hi": overall["bootstrap_ci_hi"],
                "cohens_d": overall["cohens_d"],
                "significant": overall["significant"],
                "n_features": overall["n_features"],
            })

        else:
            # Multi-level: pairwise + Holm correction
            result = compute_multilevel_significance(
                sub, factor_col=fac, levels=levels,
                metric=metric, n_bootstrap=n_bootstrap, alpha=alpha,
                correction="holm",
            )
            if result.empty:
                continue

            csv_path = os.path.join(
                output_base, f"{prefix}significance_{fac}__pairwise.csv",
            )
            result.to_csv(csv_path, index=False)

            print(f"\n[SIG] {fac}: {len(levels)} levels -> {len(result)} pairwise comparisons")
            show_cols = [c for c in ("comparison", "delta", "wilcoxon_p",
                                     "p_corrected", "significant_corrected",
                                     "cohens_d") if c in result.columns]
            print(result[show_cols].to_string(index=False))
            print(f"  Wrote {csv_path}")

            for _, row in result.iterrows():
                all_summaries.append({
                    "factor": fac,
                    "comparison": row.get("comparison"),
                    "mean_delta": row.get("delta"),
                    "wilcoxon_p": row.get("wilcoxon_p"),
                    "p_corrected": row.get("p_corrected"),
                    "ci_lo": row.get("bootstrap_ci_lo"),
                    "ci_hi": row.get("bootstrap_ci_hi"),
                    "cohens_d": row.get("cohens_d"),
                    "significant": row.get("significant_corrected", row.get("significant")),
                    "n_features": row.get("n_features"),
                })

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_path = os.path.join(output_base, f"{prefix}significance_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n[SIG] Combined summary -> {summary_path}")
        print(summary_df.to_string(index=False))
        return summary_df

    return None


def build_error_df(
    model_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    features: list[str],
    model_name: str,
    rationale_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    model_sub = safe_strip_sentence_col(model_df.copy())
    gold_sub = safe_strip_sentence_col(gold_df.copy())

    model_sub = combine_wh_qu(model_sub)
    gold_sub = combine_wh_qu(gold_sub)

    needed_feats = [f for f in features if f in model_sub.columns and f in gold_sub.columns]
    needed_cols = ["sentence"] + needed_feats

    model_sub = model_sub[needed_cols].copy()
    gold_sub = gold_sub[needed_cols].copy()

    merged = gold_sub.merge(model_sub, on="sentence", suffixes=("_gold", "_pred"))

    rat_lookup: Dict[tuple[str, str], str] = {}
    if rationale_df is not None and "sentence" in rationale_df.columns:
        r = safe_strip_sentence_col(rationale_df.copy())
        for feat in needed_feats:
            if feat in r.columns:
                rat_lookup.update(
                    {
                        (sent, feat): (txt if pd.notna(txt) else "")
                        for sent, txt in zip(r["sentence"].tolist(), r[feat].tolist())
                    }
                )

    rows = []
    for _, row in merged.iterrows():
        sent = row["sentence"]
        for feat in needed_feats:
            gold_col = f"{feat}_gold"
            pred_col = f"{feat}_pred"
            gv = row.get(gold_col, pd.NA)
            pv = row.get(pred_col, pd.NA)

            if pd.isna(gv) or pd.isna(pv):
                continue

            try:
                gv = int(gv)
                pv = int(pv)
            except Exception:
                continue

            if gv != pv:
                rows.append(
                    {
                        "model": model_name,
                        "sentence": sent,
                        "feature": feat,
                        "gold": gv,
                        "pred": pv,
                        "model_rationale": rat_lookup.get((str(sent).strip(), feat), ""),
                    }
                )

    return pd.DataFrame(rows)


def auto_pairwise_deltas(
    df: pd.DataFrame,
    *,
    factors: list[str],
    output_base: str,
    prefix: str,
    metrics=("precision", "recall", "f1"),
):
    work = df.copy().dropna(subset=["feature"])

    factors = [f for f in factors if f in work.columns]

    for fac in factors:
        levels = sorted(work[fac].dropna().unique().tolist())
        if len(levels) != 2:
            continue

        if fac in PREFERRED_LEVEL_ORDER:
            pref_a, pref_b = PREFERRED_LEVEL_ORDER[fac]
            if set(levels) == {pref_a, pref_b}:
                a_level, b_level = pref_a, pref_b
            else:
                a_level, b_level = levels[0], levels[1]
        else:
            a_level, b_level = levels[0], levels[1]

        other_factors = [f for f in factors if f != fac and work[[f]].notna().to_numpy().any()]
        sub = work.dropna(subset=[fac]).copy()
        for f in other_factors:
            sub[f] = sub[f].fillna("NA")

        index_cols = ["feature"] + other_factors

        deltas = compute_pairwise_deltas(
            sub,
            index_cols=index_cols,
            factor_col=fac,
            a_level=a_level,
            b_level=b_level,
            metrics=metrics,
            require_metric="f1",
            aggfunc="mean",
        )

        out_csv = os.path.join(output_base, f"{prefix}delta_{fac}__{b_level}_minus_{a_level}__per_feature.csv")
        deltas.to_csv(out_csv, index=False)

        col = f"d_f1_{b_level}_minus_{a_level}"
        if col in deltas.columns and not deltas.empty:
            plot_delta_heatmap(
                deltas,
                value_col=col,
                title=f"{prefix} ΔF1: {b_level} − {a_level} (factor={fac})",
                save_path=os.path.join(output_base, f"{prefix}delta_{fac}__{b_level}_minus_{a_level}__F1_heatmap.png"),
            )

        print(f"[INFO] Wrote {out_csv} ({len(deltas)} rows)")


def safe_sheet_name(name: str, max_len: int = 31) -> str:
    bad_chars = ["[", "]", "*", "?", "/", "\\", ":"]
    for ch in bad_chars:
        name = name.replace(ch, "_")
    return name[:max_len]


def _ensure_numeric_feature_set(factors: dict, detected_feature_set: Optional[str]) -> dict:
    """
    Patch: guarantee factors["feature_set"] is "17" or "25" when possible,
    even if the sheet name doesn't include it.
    detected_feature_set: "masis" or "extended" or None
    """
    out = dict(factors)
    if out.get("feature_set") in ("17", "25"):
        return out
    if detected_feature_set == "masis":
        out["feature_set"] = "17"
    elif detected_feature_set == "extended":
        out["feature_set"] = "25"
    return out


def compute_interrater_reliability(
    gold_dfs: Dict[str, pd.DataFrame],
    features: List[str],
    output_path: str,
):
    """
    Compute pairwise interrater reliability (Cohen's kappa + % agreement)
    between all gold raters for each feature.
    """
    rater_names = list(gold_dfs.keys())
    if len(rater_names) < 2:
        print("[INFO] Only one gold rater — skipping interrater reliability.")
        return

    print(f"\n{'='*60}")
    print(f"=== Interrater Reliability ({len(rater_names)} raters) ===")
    print(f"{'='*60}")

    rows = []
    for r1, r2 in combinations(rater_names, 2):
        df1 = gold_dfs[r1]
        df2 = gold_dfs[r2]

        # Align on shared sentences
        merged = df1.merge(df2, on="sentence", suffixes=(f"_{r1}", f"_{r2}"))
        n_shared = len(merged)
        if n_shared == 0:
            print(f"[WARN] No overlapping sentences between {r1} and {r2}.")
            continue

        print(f"\n{r1} vs {r2}: {n_shared} shared sentences")

        for feat in features:
            col1 = f"{feat}_{r1}"
            col2 = f"{feat}_{r2}"
            if col1 not in merged.columns or col2 not in merged.columns:
                continue

            y1 = merged[col1].dropna().astype(int)
            y2 = merged[col2].dropna().astype(int)

            # Align indices after dropna
            common = y1.index.intersection(y2.index)
            y1, y2 = y1.loc[common], y2.loc[common]

            if len(y1) == 0:
                continue

            pct_agree = (y1 == y2).mean()

            # Cohen's kappa (handle edge case: all same label)
            try:
                kappa = cohen_kappa_score(y1, y2)
            except Exception:
                kappa = float('nan')

            rows.append({
                "rater_1": r1,
                "rater_2": r2,
                "feature": feat,
                "n_sentences": len(y1),
                "pct_agreement": round(pct_agree, 4),
                "cohens_kappa": round(kappa, 4) if not np.isnan(kappa) else None,
            })

            print(f"  {feat}: kappa={kappa:.3f}  agree={pct_agree:.1%}")

    if not rows:
        print("[WARN] No interrater reliability data computed.")
        return

    irr_df = pd.DataFrame(rows)

    # Summary: average across features per pair
    summary = irr_df.groupby(["rater_1", "rater_2"]).agg(
        mean_kappa=("cohens_kappa", "mean"),
        mean_agreement=("pct_agreement", "mean"),
        n_features=("feature", "count"),
    ).reset_index()

    print(f"\n=== Interrater Reliability Summary ===")
    print(summary.to_string(index=False))

    # Save
    irr_path = os.path.join(output_path, "interrater_reliability.csv")
    irr_df.to_csv(irr_path, index=False)
    print(f"\n[INFO] Wrote interrater reliability to {irr_path}")

    summary_path = os.path.join(output_path, "interrater_reliability_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[INFO] Wrote summary to {summary_path}")


def evaluate_sheets(file_path: str, gold_dfs: Optional[Dict[str, pd.DataFrame]] = None):
    print(f"\n===============================")
    print(f"=== Evaluating {file_path} ===")
    print(f"===============================\n")

    file_basename = os.path.splitext(os.path.basename(file_path))[0]
    output_base = os.path.join(output_dir, file_basename)
    os.makedirs(output_base, exist_ok=True)

    sheets = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")

    # Load primary gold: from CLI files (first rater) or from workbook
    if gold_dfs:
        primary_rater = list(gold_dfs.keys())[0]
        gold_df = gold_dfs[primary_rater]
        print(f"[INFO] Using primary gold from CLI: {primary_rater} ({len(gold_df)} sentences)")
    else:
        gold_df = sheets.get("Gold")
        if gold_df is None:
            print(f"[WARN] No 'Gold' sheet in {file_path} and no --gold provided. Skipping.")
            return

        gold_df = drop_features_column(gold_df)
        gold_df = normalize_sentence_column(gold_df, "Gold")
        if gold_df is None:
            print(f"[WARN] Gold sheet missing sentence column in {file_path}. Skipping workbook.")
            return
        gold_df = gold_df.dropna(subset=["sentence"])
        gold_df = safe_strip_sentence_col(gold_df)

        gold_df = combine_wh_qu(gold_df)
        gold_df = gold_df.drop_duplicates(subset="sentence")

    model_sheets: dict[str, pd.DataFrame] = {}
    rationale_sheets: dict[str, pd.DataFrame] = {}

    # Determine which sheet names to skip (gold-related)
    gold_sheet_names = {"Gold"}
    if gold_dfs:
        # Also skip any sheets matching gold rater names
        gold_sheet_names.update(gold_dfs.keys())

    for sheet_name, df_raw in sheets.items():
        if sheet_name in gold_sheet_names:
            continue
        if str(sheet_name).startswith("~$"):
            continue

        sname = str(sheet_name)

        # 1. Full "_rationales" sheets (old convention)
        if sname.endswith("_rationales"):
            base_name = sname[:-11]
            rationale_sheets[base_name] = df_raw
            continue

        # 2. Any rationale-only sheet whose suffix starts with "_rats_"
        #    e.g. "_rats_r", "_rats_rat", "_rats_rati", "_rats_rationales", etc.
        if re.search(r"_rats_[A-Za-z]*$", sname):
            print(f"[INFO] Skipping rationale-only sheet {sheet_name}")
            continue


        name_up = str(sheet_name).upper()
        KNOWN_MODEL_PREFIXES = (
            "GPT", "BERT", "PHI", "MODERNBERT", "MODERN-BERT",
            "GEM", "GEMINI", "QWEN", "QWQ", "LLAMA", "TEST_",
        )
        if not any(name_up.startswith(p) for p in KNOWN_MODEL_PREFIXES):
            continue

        df = drop_features_column(df_raw)
        df = normalize_sentence_column(df, sheet_name)
        if df is None:
            continue
        df = df.dropna(subset=["sentence"])
        df = safe_strip_sentence_col(df)

        # Use model_name column (from combine_predictions.py) as the canonical name
        # This preserves the full model identifier even when prefix is stripped from sheet tab
        if "model_name" in df.columns:
            mn = str(df["model_name"].iloc[0])
            df = df.drop(columns=["model_name"])
            # Check if model_name already contains config info (ZS/FS/ZSCOT/FSCOT)
            # If it's just a bare model prefix (e.g. "GEMINI"), combine with sheet tab name
            mn_up = mn.upper()
            has_config = any(tok in mn_up.split("_") for tok in ("ZS", "FS", "ZSCOT", "FSCOT"))
            if has_config:
                canonical_name = mn
            else:
                # model_name is just the prefix — combine with sheet tab to get full config name
                sn = str(sheet_name)
                # Avoid duplicating the prefix if sheet already starts with it
                if sn.upper().startswith(mn_up):
                    canonical_name = sn
                else:
                    canonical_name = f"{mn}_{sn}"
        else:
            canonical_name = str(sheet_name)

        df = combine_wh_qu(df)
        df = df.drop_duplicates(subset="sentence")

        model_sheets[canonical_name] = df

    if not model_sheets:
        print(f"[WARN] No model sheets found in {file_path}. Expected sheets starting with model name or instruction type prefix.")
        return
    
    gold_sents = set(gold_df['sentence'].dropna())
    initial_gold_count = len(gold_sents)
    print(f"DEBUG: Starting with {initial_gold_count} Gold sentences.")

    # Keep per-model detected feature-set num to ensure factors are filled later
    model_feature_set_num: Dict[str, str] = {}

    # ==========================================
    # PASS 1: Per-model evaluation (each model on all its completed sentences vs Gold)
    # ==========================================
    print("\n" + "="*60)
    print("PASS 1: Per-model evaluation (all completed sentences)")
    print("="*60)

    permodel_eval_results: list[pd.DataFrame] = []
    per_model_errors: dict[str, pd.DataFrame] = {}
    permodel_output = os.path.join(output_base, "per_model")
    os.makedirs(permodel_output, exist_ok=True)

    for sheet_name, model_df in model_sheets.items():
        detected = detect_feature_set(sheet_name, model_df)
        if detected is None:
            print(f"[SKIP] {sheet_name}: could not determine feature set.")
            continue

        features = MASIS17_FEATURES if detected == "masis" else EXTENDED_FEATURES
        model_feature_set_num[sheet_name] = ("17" if detected == "masis" else "25")

        model_df = binarize_if_probabilistic(model_df, features, str(sheet_name))
        model_sheets[sheet_name] = model_df

        # Per-model: intersect only this model with Gold
        model_sents = set(model_df['sentence'].dropna())
        shared_with_gold = gold_sents.intersection(model_sents)
        n_completed = len(shared_with_gold)
        print(f"\n--- [Per-model] {sheet_name}: {n_completed}/{initial_gold_count} sentences (feature set: {detected}) ---")

        if not shared_with_gold:
            print(f"[SKIP] {sheet_name}: no sentences overlap with Gold.")
            continue

        gold_subset = gold_df[gold_df['sentence'].isin(shared_with_gold)].copy()
        model_subset = model_df[model_df['sentence'].isin(shared_with_gold)].copy()

        model_eval = evaluate_model(
            model_df=model_subset,
            truth_df=gold_subset,
            model_name=str(sheet_name),
            features=features,
            output_base=permodel_output,
        )

        if not model_eval.empty:
            model_eval["n_sentences"] = n_completed
            permodel_eval_results.append(model_eval)

            err_df = build_error_df(
                model_df=model_subset,
                gold_df=gold_subset,
                features=features,
                model_name=str(sheet_name),
                rationale_df=rationale_sheets.get(sheet_name),
            )
            per_model_errors[str(sheet_name)] = err_df

            rat_df = rationale_sheets.get(sheet_name)
            if rat_df is not None:
                try:
                    annotated = build_annotated_rationales(
                        pred_df=model_subset,
                        rationale_df=rat_df,
                        truth_df=gold_subset,
                        features=features,
                        only_disagreements=True,
                        max_rows=None,
                    )
                    out_csv = os.path.join(permodel_output, f"{sheet_name}_annotated_rationales.csv")
                    annotated.to_csv(out_csv, index=False)
                    print(f"[INFO] Wrote annotated rationales for {sheet_name} to {out_csv}")
                except Exception as e:
                    print(f"[WARN] Could not build annotated rationales for {sheet_name}: {e}")

    # Save per-model summary CSV
    if permodel_eval_results:
        permodel_all = pd.concat(permodel_eval_results, ignore_index=True)
        permodel_all.to_csv(os.path.join(permodel_output, "all_per_model_results.csv"), index=False)
        print(f"\n[INFO] Per-model results saved to {permodel_output}/all_per_model_results.csv")

    # ==========================================
    # PASS 2: Strict intersection (apples-to-apples comparison)
    # ==========================================
    print("\n" + "="*60)
    print("PASS 2: Strict intersection (shared sentences across ALL models)")
    print("="*60)

    common_ids = set(gold_sents)
    for name, df in model_sheets.items():
        model_sents = set(df['sentence'].dropna())
        before_count = len(common_ids)
        common_ids = common_ids.intersection(model_sents)
        dropped = before_count - len(common_ids)
        if dropped > 0:
            print(f"DEBUG: Dropping {dropped} sentences missing from model '{name}' (Retaining {len(common_ids)})")

    if not common_ids:
        print("[WARN] Intersection of sentence IDs is EMPTY. Skipping strict comparison.")
        print("[INFO] Per-model results are still available above.")
        # Use per-model results for downstream plots
        eval_results = permodel_eval_results
    else:
        if len(common_ids) < initial_gold_count:
            print(f"STRICT MODE: Pruned evaluation set from {initial_gold_count} to {len(common_ids)} shared sentences.")

        gold_strict = gold_df[gold_df['sentence'].isin(common_ids)].copy()

        eval_results: list[pd.DataFrame] = []

        for sheet_name, model_df in model_sheets.items():
            if sheet_name not in model_feature_set_num:
                continue

            detected_num = model_feature_set_num[sheet_name]
            features = MASIS17_FEATURES if detected_num == "17" else EXTENDED_FEATURES

            model_strict = model_df[model_df['sentence'].isin(common_ids)].copy()

            print(f"\n--- [Strict] {sheet_name}: {len(common_ids)} shared sentences ---")

            model_eval = evaluate_model(
                model_df=model_strict,
                truth_df=gold_strict,
                model_name=str(sheet_name),
                features=features,
                output_base=output_base,
            )

            if not model_eval.empty:
                model_eval["n_sentences"] = len(common_ids)
                eval_results.append(model_eval)

    eval_results = [df for df in eval_results if not df.empty]
    if not eval_results:
        print("[WARN] No non-empty evaluation results.")
        return


    all_eval = pd.concat(eval_results, ignore_index=True)

    # Add factor columns (patched: ensure feature_set num exists even when name doesn't contain it)
    def _factors_for_model_name(mname: str) -> dict:
        base = parse_factors(mname)
        # if parse fails for feature_set, backfill from detected
        detected_num = model_feature_set_num.get(mname)
        if base.get("feature_set") is None and detected_num in ("17", "25"):
            base["feature_set"] = detected_num
        return base

    factors_df = all_eval["model"].apply(lambda x: pd.Series(_factors_for_model_name(str(x))))
    factors_df = factors_df.loc[:, ~factors_df.columns.duplicated()]

    # avoid collisions: drop any existing factor columns before concat
    for c in factors_df.columns:
        if c in all_eval.columns:
            all_eval = all_eval.drop(columns=[c])

    all_eval = pd.concat([all_eval, factors_df], axis=1)

    # GPT-only deltas
    gpt_eval = all_eval[all_eval["model"].str.upper().str.startswith("GPT_")].copy()
    gpt_eval = gpt_eval.dropna(subset=["feature_set", "ctx"])

    auto_pairwise_deltas(
        gpt_eval,
        factors=["feature_set", "ctx", "instr", "leg", "trial"],
        output_base=output_base,
        prefix="GPT_",
    )

    # all-model deltas (only where ctx and leg exist)
    auto_pairwise_deltas(
        all_eval.dropna(subset=["ctx"]),
        factors=["feature_set", "ctx", "instr", "leg"],
        output_base=output_base,
        prefix="ALLMODELS_",
    )

    # ==========================================
    # Significance tests + summary visualizations
    # ==========================================

    # GPT-only significance tests
    gpt_sig_df = None
    if not gpt_eval.empty:
        gpt_sig_df = auto_significance_tests(
            gpt_eval,
            factors=["feature_set", "ctx", "instr", "leg", "trial"],
            output_base=output_base,
            prefix="GPT_",
        )

    # All-model significance tests
    all_sig_df = auto_significance_tests(
        all_eval.dropna(subset=["ctx"]),
        factors=["feature_set", "ctx", "instr", "leg"],
        output_base=output_base,
        prefix="ALLMODELS_",
    )

    # Forest plots for variable effects
    if gpt_sig_df is not None:
        plot_variable_effects(
            gpt_sig_df,
            save_path=os.path.join(output_base, "GPT_variable_effects_forest.png"),
        )

    if all_sig_df is not None:
        plot_variable_effects(
            all_sig_df,
            save_path=os.path.join(output_base, "ALLMODELS_variable_effects_forest.png"),
        )

    # Config leaderboard
    plot_config_leaderboard(
        all_eval,
        save_path=os.path.join(output_base, "config_leaderboard.png"),
    )

    # Summary table CSV
    generate_summary_table(
        all_eval,
        save_path=os.path.join(output_base, "summary_leaderboard.csv"),
    )

    # Dedicated plots for 25-only
    ext_eval = all_eval[all_eval["feature_set"] == "25"].copy()
    if not ext_eval.empty:
        ext_models = ext_eval["model"].unique().tolist()
        ext_dfs = [df for df in eval_results if df["model"].iloc[0] in ext_models]

        plot_model_metrics(
            eval_dfs=ext_dfs,
            metric="f1",
            style="heatmap",
            align="intersection",
            title="EXTENDED (25): F1 by feature (shared across 25-feature models)",
            save_path=os.path.join(output_base, "EXT25_F1_heatmap_intersection.png"),
            figsize=(14, 10),
        )

    # Dedicated plots for 17-only
    mas_eval = all_eval[all_eval["feature_set"] == "17"].copy()
    if not mas_eval.empty:
        mas_models = mas_eval["model"].unique().tolist()
        mas_dfs = [df for df in eval_results if df["model"].iloc[0] in mas_models]

        plot_model_metrics(
            eval_dfs=mas_dfs,
            metric="f1",
            style="heatmap",
            align="intersection",
            title="MASIS (17): F1 by feature (shared across 17-feature models)",
            save_path=os.path.join(output_base, "MAS17_F1_heatmap_intersection.png"),
            figsize=(14, 10),
        )

    # GPT-only means/vars + manual deltas
    if gpt_eval.empty:
        print("[WARN] No GPT_* model rows found; skipping GPT-only analyses (means/vars/deltas).")
    else:
        mean_f1_per_feature = (
            gpt_eval.groupby("feature")["f1"].mean().sort_values(ascending=False).reset_index(name="mean_f1_gpt")
        )
        mean_f1_per_feature.to_csv(os.path.join(output_base, "gpt_mean_f1_per_feature.csv"), index=False)

        var_f1_per_feature = gpt_eval.groupby("feature")["f1"].var(ddof=1).reset_index(name="var_f1_gpt")
        var_f1_per_feature.to_csv(os.path.join(output_base, "gpt_var_f1_per_feature.csv"), index=False)

        # shared features across GPT models
        shared = None
        for _, sub in gpt_eval.groupby("model"):
            feats = set(sub["feature"].unique())
            shared = feats if shared is None else (shared & feats)
        shared = shared or set()
        gpt_shared = gpt_eval[gpt_eval["feature"].isin(shared)].copy()

        delta_instr = compute_pairwise_deltas(
            gpt_shared,
            index_cols=["feature", "feature_set"],
            factor_col="instr",
            a_level="ZS",
            b_level="FSCOT",
            metrics=("precision", "recall", "f1"),
            require_metric="f1",
            aggfunc="mean",
        )
        delta_instr.to_csv(os.path.join(output_base, "delta_ZS_to_FSCOT_per_feature.csv"), index=False)

        delta_ctx = compute_pairwise_deltas(
            gpt_shared,
            index_cols=["feature", "feature_set", "instr"],
            factor_col="ctx",
            a_level="noCTX",
            b_level="CTX",
            metrics=("precision", "recall", "f1"),
            require_metric="f1",
            aggfunc="mean",
        )

        out_csv = os.path.join(output_base, "delta_CTX_per_feature.csv")
        delta_ctx.to_csv(out_csv, index=False)
        print(f"[INFO] Wrote {out_csv} ({len(delta_ctx)} rows)")

        # FIX: correct column name
        colname = "d_f1_CTX_minus_noCTX"
        if colname in delta_ctx.columns:
            plot_delta_heatmap(
                delta_ctx,
                value_col=colname,
                title="GPT ΔF1: CTX − noCTX (per feature; paired within other factors)",
                save_path=os.path.join(output_base, "GPT_delta_CTX_f1_heatmap.png"),
            )
        else:
            print(f"[WARN] Expected {colname} not found in delta_ctx columns: {list(delta_ctx.columns)}")

    # Cross-model comparisons
    all_models_label = "ALL_MODELS"

    plot_model_metrics(
        eval_dfs=eval_results,
        metric="f1",
        style="heatmap",
        align="intersection",
        title="Model F1 by AAE feature (heatmap, shared features only)",
        save_path=os.path.join(output_base, f"{all_models_label}_F1_heatmap_intersection.png"),
    )

    plot_overall_f1_scores(
        eval_dfs=eval_results,
        align="intersection",
        save_path=os.path.join(output_base, f"{all_models_label}_Overall_F1_intersection.png"),
    )

    plot_overall_f1_scores(
        eval_dfs=eval_results,
        align="union",
        save_path=os.path.join(output_base, f"{all_models_label}_Overall_F1_full_union.png"),
    )

    plot_f1_scores_per_feature(
        eval_dfs=eval_results,
        align="intersection",
        save_path=os.path.join(output_base, f"{all_models_label}_Per_Feature_F1_intersection.png"),
    )

    plot_overall_micro_f1_scores(
        eval_dfs=eval_results,
        align="intersection",
        save_path=os.path.join(output_base, f"{all_models_label}_MicroF1_intersection.png"),
    )

    plot_overall_micro_f1_scores(
        eval_dfs=eval_results,
        align="union",
        save_path=os.path.join(output_base, f"{all_models_label}_MicroF1_full_union.png"),
    )

    # Error aggregation + Excel
    all_errors = pd.concat(per_model_errors.values(), ignore_index=True) if per_model_errors else pd.DataFrame()
    if not all_errors.empty:
        err_counts = (
            all_errors.groupby(["model", "feature"]).size().reset_index(name="errors")
            .pivot(index="feature", columns="model", values="errors")
            .fillna(0)
            .sort_index()
        )

        print("\n=== Error counts (per feature x model) ===")
        print(err_counts)

        plot_model_metrics(
            err_counts=err_counts.reset_index().melt(id_vars="feature", var_name="model", value_name="errors"),
            metric="errors",
            style="heatmap",
            title="Errors per Feature by Model",
            annotate_heatmap=True,
            figsize=(12, 8),
            save_path=os.path.join(output_base, f"{all_models_label}_Error_Heatmap.png"),
        )

        errors_xlsx = os.path.join(output_base, "model_errors_all_experiments.xlsx")
        with pd.ExcelWriter(errors_xlsx) as writer:
            for model_name, err_df in per_model_errors.items():
                if err_df.empty:
                    continue
                sheet = safe_sheet_name(f"{model_name}_errors")
                err_df.to_excel(writer, sheet_name=sheet, index=False)

            all_errors.to_excel(writer, sheet_name="all_errors", index=False)
            err_counts.to_excel(writer, sheet_name="error_counts_pivot")

        print(f"[INFO] Wrote error breakdown to {errors_xlsx}")

    # Interrater reliability (if multiple gold raters provided)
    if gold_dfs and len(gold_dfs) >= 2:
        # Determine features from the first model sheet
        irr_features = list(EXTENDED_FEATURES) if any(
            f in gold_df.columns for f in EXTENDED_FEATURES if f not in MASIS17_FEATURES
        ) else list(MASIS17_FEATURES)
        compute_interrater_reliability(gold_dfs, irr_features, output_base)


def load_gold_file(path: str) -> pd.DataFrame:
    """Load a gold labels file (CSV or Excel) and normalize it."""
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    else:
        # Excel: read first sheet (or 'Gold' sheet if it exists)
        sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
        if "Gold" in sheets:
            df = sheets["Gold"]
        else:
            # Use first sheet
            df = list(sheets.values())[0]

    df = drop_features_column(df)
    df = normalize_sentence_column(df, os.path.basename(path))
    if df is None:
        raise ValueError(f"Gold file {path} missing sentence column.")
    df = df.dropna(subset=["sentence"])
    df = safe_strip_sentence_col(df)
    df = combine_wh_qu(df)
    df = df.drop_duplicates(subset="sentence")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate AAE Model Predictions against Gold Standard."
    )

    parser.add_argument(
        "input_path",
        nargs="?",
        default="data/*.xlsx",
        help="Path to a specific .xlsx file, a directory, or a glob pattern (default: data/*.xlsx)"
    )

    parser.add_argument(
        "--gold", nargs="+", default=None,
        help="Gold label file(s) as CSV or Excel. First file is primary gold. "
             "Additional files are secondary raters for interrater reliability. "
             "If omitted, reads 'Gold' sheet from each input workbook."
    )

    parser.add_argument(
        "--output-dir",
        default="Results/",
        help="Base directory to save evaluation results"
    )

    args = parser.parse_args()

    # 1. Update the global output directory variable used by evaluate_sheets
    global output_dir
    output_dir = args.output_dir

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        print(f"[INFO] Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    # 2. Load gold files if provided via CLI
    gold_dfs = None
    if args.gold:
        gold_dfs = {}
        for gpath in args.gold:
            rater_name = os.path.splitext(os.path.basename(gpath))[0]
            print(f"[INIT] Loading gold file: {gpath} (rater: {rater_name})")
            gold_dfs[rater_name] = load_gold_file(gpath)
        print(f"[INIT] Loaded {len(gold_dfs)} gold rater(s): {list(gold_dfs.keys())}")

    # 3. Resolve input paths (handle files, directories, and glob patterns)
    filepaths = []

    # Check if input is a direct directory
    if os.path.isdir(args.input_path):
        filepaths = glob.glob(os.path.join(args.input_path, "*.xlsx"))
    # Check if input is a glob pattern or specific file
    else:
        filepaths = glob.glob(args.input_path)

    # Filter out temporary Excel files (starting with ~$)
    filepaths = [fp for fp in filepaths if not os.path.basename(fp).startswith("~$")]

    if not filepaths:
        print(f"[ERROR] No valid .xlsx files found matching: {args.input_path}")
        sys.exit(1)

    print(f"\n[INIT] Found {len(filepaths)} file(s) to evaluate in '{output_dir}':")
    for fp in filepaths:
        print(f" - {fp}")

    # 4. Run evaluation for each file
    for filepath in filepaths:
        try:
            evaluate_sheets(filepath, gold_dfs=gold_dfs)
        except Exception as e:
            print(f"\n[ERROR] Failed to evaluate {filepath}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
