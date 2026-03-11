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
try:
    import statsmodels.formula.api as _smf
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

# Feature lists inlined to avoid importing heavy model dependencies from multi_prompt_configs
MASIS_FEATURES = [
    "zero-poss", "zero-copula", "double-tense", "be-construction", "resultant-done",
    "finna", "come", "double-modal", "multiple-neg", "neg-inversion", "n-inv-neg-concord",
    "aint", "zero-3sg-pres-s", "is-was-gen", "zero-pl-s", "double-object", "wh-qu1", "wh-qu2",
]
EXTENDED_FEATURES = MASIS_FEATURES + [
    "existential-it", "demonstrative-them", "appositive-pleonastic-pronoun",
    "bin", "verb-stem", "past-tense-swap", "zero-rel-pronoun", "preterite-had", "bare-got",
]

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
    "cot": ("noCoT", "CoT"),          # CoT - noCoT
}

# one definition only
feat_thresholds = {
    "multiple-neg": 0.5,
}

# Maps lowercase model name prefixes (from sheet/filename) → display names for plots
# More specific prefixes must come first (dict iteration order = insertion order)
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "gemini3-flash-thinking": "Gemini 3 Flash Thinking",
    "gemini3-flash": "Gemini 3 Flash",
    "gemini3_pro": "Gemini 3 Pro",
    "gemini": "Gemini 2.5 Flash",
    "gem": "Gemini 2.5 Flash",
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
      New style: PHI_ZS_noCTX_noLeg / GEMINI_FScot_CTX5_Leg / PHI4_ZScot_CTX1t_noLeg
    Returns dict with keys:
      feature_set in {"17","25"} or None
      instr in {"ZS","FS","ZScot","FScot"} or None
      ctx in {"noCTX","CTX1t","CTX5"} or None
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
    elif "CTX5" in toks or "CTXWIDE" in toks:
        ctx = "CTX5"
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

def prettify_model_label(name: str) -> str:
    """
    Replace raw model name prefix with a display-friendly label using MODEL_DISPLAY_NAMES.
    E.g. "GEMINI_ZS_noCTX_noLeg" → "Gemini 2.5 Flash_ZS_noCTX_noLeg"
         "Gemini"                 → "Gemini 2.5 Flash"
    """
    lower = str(name).lower()
    for raw, display in MODEL_DISPLAY_NAMES.items():
        if lower.startswith(raw):
            suffix = name[len(raw):]
            return display + suffix
    return name


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
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0.0,
                annot_kws={"size": 7}, linewidths=0.3, linecolor="white")
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
    title: Optional[str] = None,
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
    ax.set_title(title or (
        "Overall micro-F1 by model (shared features only)"
        if align == "intersection"
        else "Overall micro-F1 by model (each model’s evaluated features; not strictly comparable)"
    ))
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
    title: Optional[str] = None,
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
    ax.set_title(title or (
        "Overall F1 Scores by Model (shared features only)"
        if align == "intersection"
        else "Overall F1 Scores by Model (each model’s evaluated features; not strictly comparable)"
    ))
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
    title: Optional[str] = None,
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
    title = title or ("F1 Scores per Feature by Model (Shared Features Only)" if align == "intersection" else "F1 Scores per Feature by Model")
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
            annot_kws={"size": 7},
            linewidths=0.3,
            linecolor="white",
        )
        plt.title(title)
        plt.ylabel(ylab)
        plt.xlabel(xlab)
        plt.xticks(rotation=90)
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
      PHI_ZS_noCTX_noLeg, GEMINI_FScot_CTX5_Leg, PHI4_ZScot_CTX1t_noLeg
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
    elif "CTX5" in base or "CTXWIDE" in base:
        ctx = "Context (5-turn)"
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
    title: Optional[str] = None,
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
        cmap = plt.colormaps["Set2"].resampled(max(len(unique_instr), 1))        
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
    ax.set_title(title or "Configuration Leaderboard (ranked by macro-F1)")
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
    title: Optional[str] = None,
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
    ax.set_title(title or "Variable Effects on F1 (forest plot)")

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


def plot_config_impact_panel(
    all_eval: pd.DataFrame,
    output_base: str,
    prefix: str,
    factors: Optional[List[str]] = None,
    metric: str = "f1",
    model_label: Optional[str] = None,
    figsize_per_factor: tuple = (3.5, 4.0),
) -> None:
    """
    Pairwise config-effect panel: for each config dimension (ctx, instr, leg,
    feature_set), show mean ΔF1 (b − a) for every model as a horizontal bar.
    Green = helps, red = hurts.  One subplot per factor, models on Y-axis.

    Direction follows PREFERRED_LEVEL_ORDER (e.g. CTX − noCTX, FScot − ZS).
    Saves: {prefix}config_impact_panel.png
    """
    if factors is None:
        factors = ["ctx", "instr", "cot", "leg", "feature_set"]

    df = all_eval.copy().dropna(subset=["feature", metric])

    # Derive CoT column BEFORE collapsing instr, so CoT info isn't lost.
    # ZScot / FScot → "CoT";  ZS / FS → "noCoT"
    if "instr" in df.columns:
        df["cot"] = df["instr"].map(
            lambda x: "CoT" if str(x) in ("ZScot", "FScot") else ("noCoT" if pd.notna(x) else None)
        )

    # Filter factors to those present in df (after deriving cot above)
    factors = [f for f in factors if f in df.columns]
    if not factors:
        print("[INFO] plot_config_impact_panel: no factor columns found; skipping.")
        return

    # Collapse multi-level factors to binary so all can be plotted:
    #   ctx:   CTX1t / CTX5  → CTX  (any context vs none)
    #   instr: ZScot → ZS,  FScot → FS  (examples vs no-examples)
    if "ctx" in df.columns:
        df["ctx"] = df["ctx"].replace({"CTX1t": "CTX", "CTX5": "CTX"})
    if "instr" in df.columns:
        df["instr"] = df["instr"].replace({"ZScot": "ZS", "FScot": "FS"})

    # Extract base model for grouping. For single-model workbooks the sheet names
    # don't contain the model name (e.g. "FS_CTX1t_Leg"), so parse_model_config
    # misidentifies them. Use model_label override when provided.
    if model_label:
        df["_base"] = model_label
    else:
        df["_base"] = df["model"].apply(lambda x: parse_model_config(x)[0])
    base_models = sorted(df["_base"].unique())

    factor_labels = {
        "ctx":         "Context\n(CTX − noCTX)",
        "instr":       "Examples\n(FS − ZS)",
        "cot":         "Chain-of-Thought\n(CoT − noCoT)",
        "leg":         "Legitimacy\n(Leg − noLeg)",
        "feature_set": "Feature Set\n(25 − 17)",
        "trial":       "Trial\n(B − A)",
    }

    # Local override: after collapsing FScot→FS / ZScot→ZS, instr levels are ZS/FS
    _panel_level_order = {**PREFERRED_LEVEL_ORDER, "instr": ("ZS", "FS")}

    n = len(factors)
    fig_w = figsize_per_factor[0] * n
    fig_h = figsize_per_factor[1] + max(0, len(base_models) - 4) * 0.35
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, fac in zip(axes, factors):
        levels = sorted(df[fac].dropna().unique().tolist())
        if len(levels) != 2:
            ax.axis("off")
            ax.set_title(factor_labels.get(fac, fac), fontsize=9)
            continue

        if fac in _panel_level_order:
            pa, pb = _panel_level_order[fac]
            a_level, b_level = (pa, pb) if set(levels) == {pa, pb} else (levels[0], levels[1])
        else:
            a_level, b_level = levels[0], levels[1]

        other_facs = [f for f in factors if f != fac]
        rows = []
        for model in base_models:
            sub = df[df["_base"] == model].dropna(subset=[fac]).copy()
            if sub.empty:
                continue
            for f in other_facs:
                if f in sub.columns:
                    sub[f] = sub[f].fillna("NA")
            idx_cols = ["feature"] + [f for f in other_facs if f in sub.columns]
            deltas = compute_pairwise_deltas(
                sub,
                index_cols=idx_cols,
                factor_col=fac,
                a_level=a_level,
                b_level=b_level,
                metrics=(metric,),
                require_metric=metric,
                aggfunc="mean",
            )
            col = f"d_{metric}_{b_level}_minus_{a_level}"
            if col not in deltas.columns or deltas.empty:
                continue
            rows.append({"model": model, "delta": float(deltas[col].mean())})

        if not rows:
            ax.axis("off")
            ax.set_title(factor_labels.get(fac, fac), fontsize=9)
            continue

        plot_df = pd.DataFrame(rows).sort_values("delta").reset_index(drop=True)
        colors = ["#2ca02c" if d >= 0 else "#d62728" for d in plot_df["delta"]]
        y_pos = list(range(len(plot_df)))

        ax.barh(y_pos, plot_df["delta"], color=colors, edgecolor="white", height=0.6, zorder=2)
        ax.axvline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.6, zorder=1)
        ax.axvspan(-0.002, 0.002, color="black", alpha=0.07, zorder=0)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df["model"].tolist(), fontsize=8)
        ax.set_xlabel(f"Mean Δ{metric.upper()}", fontsize=10)
        ax.set_title(
            f"{factor_labels.get(fac, fac)}\n({b_level} − {a_level})",
            fontsize=10, fontweight="bold",
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=9, rotation=0)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5, symmetric=True))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.2f}"))
        ax.grid(axis="x", alpha=0.2, zorder=0)

        # Annotate bar ends with delta value
        x_range = plot_df["delta"].abs().max() or 0.01
        pad = x_range * 0.05
        for i, row in plot_df.iterrows():
            ha = "left" if row["delta"] >= 0 else "right"
            x = row["delta"] + (pad if row["delta"] >= 0 else -pad)
            ax.text(x, i, f"{row['delta']:+.3f}", va="center", ha=ha, fontsize=9)

    fig.suptitle(
        f"{prefix.rstrip('_')}: Config Effects on F1  (green = helps ▲, red = hurts ▼)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    save_path = os.path.join(output_base, f"{prefix}config_impact_panel.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved config impact panel to {save_path}")


def plot_ctx_breakdown(
    all_eval: pd.DataFrame,
    output_base: str,
    prefix: str,
    metric: str = "f1",
    model_label: Optional[str] = None,
    figsize: tuple = (12, 4),
) -> None:
    """
    Compares all three ctx level pairs side-by-side:
      CTX1t − noCTX  : does 1-turn context help vs none?
      CTX5 − noCTX  : does wide context help vs none?
      CTX5 − CTX1t  : is wide context better than 1-turn?

    One subplot per comparison, bars per model.
    Green = helps, red = hurts.
    Saves: {prefix}ctx_breakdown.png
    """
    if "ctx" not in all_eval.columns:
        print("[INFO] plot_ctx_breakdown: no ctx column; skipping.")
        return

    ctx_levels = all_eval["ctx"].dropna().unique()
    # Need all three levels present to make all comparisons meaningful
    have = set(ctx_levels)
    comparisons = [
        ("noCTX", "CTX1t", "CTX1t − noCTX\n(1-turn vs none)"),
        ("noCTX", "CTX5", "CTX5 − noCTX\n(wide context vs none)"),
        ("CTX1t", "CTX5", "CTX5 − CTX1t\n(wide vs 1-turn)"),
    ]
    comparisons = [(a, b, lbl) for a, b, lbl in comparisons if a in have and b in have]
    if not comparisons:
        print("[INFO] plot_ctx_breakdown: insufficient ctx levels; skipping.")
        return

    df = all_eval.copy().dropna(subset=["feature", metric])

    if model_label:
        df["_base"] = model_label
    else:
        df["_base"] = df["model"].apply(lambda x: parse_model_config(x)[0])
    base_models = sorted(df["_base"].unique())

    other_factors = [f for f in ["instr", "leg", "feature_set"] if f in df.columns]

    n = len(comparisons)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (a_level, b_level, label) in zip(axes, comparisons):
        rows = []
        for model in base_models:
            sub = df[df["_base"] == model].dropna(subset=["ctx"]).copy()
            if sub.empty:
                continue
            for f in other_factors:
                sub[f] = sub[f].fillna("NA")
            idx_cols = ["feature"] + other_factors
            deltas = compute_pairwise_deltas(
                sub,
                index_cols=idx_cols,
                factor_col="ctx",
                a_level=a_level,
                b_level=b_level,
                metrics=(metric,),
                require_metric=metric,
                aggfunc="mean",
            )
            col = f"d_{metric}_{b_level}_minus_{a_level}"
            if col not in deltas.columns or deltas.empty:
                continue
            rows.append({"model": model, "delta": float(deltas[col].mean())})

        if not rows:
            ax.axis("off")
            ax.set_title(label, fontsize=10)
            continue

        plot_df = pd.DataFrame(rows).sort_values("delta").reset_index(drop=True)
        colors = ["#2ca02c" if d >= 0 else "#d62728" for d in plot_df["delta"]]
        y_pos = list(range(len(plot_df)))

        ax.barh(y_pos, plot_df["delta"], color=colors, edgecolor="white", height=0.6, zorder=2)
        ax.axvline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.6, zorder=1)
        ax.axvspan(-0.002, 0.002, color="black", alpha=0.07, zorder=0)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df["model"].tolist(), fontsize=9)
        ax.set_xlabel(f"Mean Δ{metric.upper()}", fontsize=10)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=9, rotation=0)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5, symmetric=True))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.2f}"))
        ax.grid(axis="x", alpha=0.2, zorder=0)

        x_range = plot_df["delta"].abs().max() or 0.01
        pad = x_range * 0.05
        for i, row in plot_df.iterrows():
            ha = "left" if row["delta"] >= 0 else "right"
            x = row["delta"] + (pad if row["delta"] >= 0 else -pad)
            ax.text(x, i, f"{row['delta']:+.3f}", va="center", ha=ha, fontsize=9)

    fig.suptitle(
        f"{prefix.rstrip('_')}: Context Depth Breakdown  (green = helps ▲, red = hurts ▼)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    save_path = os.path.join(output_base, f"{prefix}ctx_breakdown.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved ctx breakdown to {save_path}")


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
    return pd.Series(index=pivot.index, data=np.nan, dtype="float64")


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
    order = np.argsort(np.where(np.isnan(p_values), np.inf, p_values))
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
    avg_human_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute pairwise interrater reliability (Cohen's kappa + % agreement)
    between all gold raters for each feature.
    If avg_human_df is provided, also computes each rater vs average human.
    Returns a DataFrame of each rater's kappa vs average human (empty if not provided).
    """
    rater_names = list(gold_dfs.keys())
    if len(rater_names) < 2:
        print("[INFO] Only one gold rater — skipping interrater reliability.")
        return pd.DataFrame()  # instead of bare return

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


            y1 = pd.to_numeric(merged[col1], errors="coerce").dropna().astype(int)
            y2 = pd.to_numeric(merged[col2], errors="coerce").dropna().astype(int)

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
        return pd.DataFrame()  # instead of bare return

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

    # Compute each rater vs average human
    human_vs_avg_rows = []
    if avg_human_df is not None:
        feat_cols = [f for f in features if f in avg_human_df.columns]
        avg_sub = avg_human_df[["sentence"] + feat_cols].rename(
            columns={f: f"{f}__avg" for f in feat_cols}
        )
        for rater, df in gold_dfs.items():
            rater_sub = df[["sentence"] + [f for f in feat_cols if f in df.columns]].rename(
                columns={f: f"{f}__rater" for f in feat_cols if f in df.columns}
            )
            merged = rater_sub.merge(avg_sub, on="sentence", how="inner")
            if merged.empty:
                continue
            for feat in feat_cols:
                col_r, col_a = f"{feat}__rater", f"{feat}__avg"
                if col_r not in merged.columns or col_a not in merged.columns:
                    continue

                y_r = pd.to_numeric(merged[col_r], errors="coerce").dropna().astype(int)
                y_a = pd.to_numeric(merged[col_a], errors="coerce").dropna().astype(int)
                common = y_r.index.intersection(y_a.index)
                y_r, y_a = y_r.loc[common], y_a.loc[common]
                if len(y_r) == 0:
                    continue
                pct = (y_r == y_a).mean()
                try:
                    kappa = cohen_kappa_score(y_r, y_a)
                except Exception:
                    kappa = float("nan")
                human_vs_avg_rows.append({
                    "entity": rater,
                    "entity_type": "human",
                    "feature": feat,
                    "n_sentences": len(y_r),
                    "pct_agreement": round(pct, 4),
                    "cohens_kappa": round(kappa, 4) if not np.isnan(kappa) else None,
                })

    return pd.DataFrame(human_vs_avg_rows)


def compute_average_human_labels(
    gold_dfs: Dict[str, pd.DataFrame],
    features: List[str],
) -> pd.DataFrame:
    """
    Compute majority-vote average label across all human raters per sentence per feature.
    With an odd number of raters, majority = round(mean >= 0.5).
    """
    rater_names = list(gold_dfs.keys())
    feat_cols = [f for f in features if any(f in df.columns for df in gold_dfs.values())]

    base = None
    for rater, df in gold_dfs.items():
        sub = df[["sentence"] + [f for f in feat_cols if f in df.columns]].copy()
        sub = sub.rename(columns={f: f"{f}__{rater}" for f in feat_cols if f in df.columns})
        base = sub if base is None else base.merge(sub, on="sentence", how="inner")

    if base is None or base.empty:
        return pd.DataFrame(columns=["sentence"] + feat_cols)

    result = base[["sentence"]].copy()
    for feat in feat_cols:
        rater_cols = [f"{feat}__{r}" for r in rater_names if f"{feat}__{r}" in base.columns]
        if not rater_cols:
            continue
        result[feat] = (base[rater_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1) >= 0.5).astype(int)
    return result


def compute_model_kappas_vs_average(
    model_sheets: Dict[str, pd.DataFrame],
    avg_human_df: pd.DataFrame,
    features: List[str],
) -> pd.DataFrame:
    """
    For each model sheet, compute Cohen's kappa and % agreement vs average human labels.
    """
    rows = []
    feat_cols = [f for f in features if f in avg_human_df.columns]
    avg_sub = avg_human_df[["sentence"] + feat_cols].rename(
        columns={f: f"{f}__avg" for f in feat_cols}
    )
    for model_name, model_df in model_sheets.items():
        model_sub = model_df[["sentence"] + [f for f in feat_cols if f in model_df.columns]].rename(
            columns={f: f"{f}__model" for f in feat_cols if f in model_df.columns}
        )
        merged = model_sub.merge(avg_sub, on="sentence", how="inner")
        if merged.empty:
            continue
        for feat in feat_cols:
            col_m, col_a = f"{feat}__model", f"{feat}__avg"
            if col_m not in merged.columns or col_a not in merged.columns:
                continue
            y_m = pd.to_numeric(merged[col_m], errors="coerce").dropna().astype(int)
            y_a = pd.to_numeric(merged[col_a], errors="coerce").dropna().astype(int)
            common = y_m.index.intersection(y_a.index)
            y_m, y_a = y_m.loc[common], y_a.loc[common]
            if len(y_m) == 0:
                continue
            pct = (y_m == y_a).mean()
            try:
                kappa = cohen_kappa_score(y_m, y_a)
            except Exception:
                kappa = float("nan")
            rows.append({
                "entity": model_name,
                "entity_type": "model",
                "feature": feat,
                "n_sentences": len(y_m),
                "pct_agreement": round(pct, 4),
                "cohens_kappa": round(kappa, 4) if not np.isnan(kappa) else None,
            })
    return pd.DataFrame(rows)


def write_irr_model_comparison(
    human_kappas: pd.DataFrame,
    model_kappas: pd.DataFrame,
    output_path: str,
):
    """
    Write a combined comparison: each human rater vs average human,
    alongside each model configuration vs average human.
    """
    combined = pd.concat([human_kappas, model_kappas], ignore_index=True)
    if combined.empty:
        print("[INFO] No IRR vs average human data to write.")
        return

    summary = (
        combined.groupby(["entity", "entity_type"])
        .agg(
            mean_kappa_vs_avg_human=("cohens_kappa", "mean"),
            mean_pct_agreement_vs_avg_human=("pct_agreement", "mean"),
            n_features=("feature", "count"),
        )
        .reset_index()
        .sort_values(["entity_type", "mean_kappa_vs_avg_human"], ascending=[True, False])
    )

    print(f"\n{'='*70}")
    print("=== Agreement with Average Human: Raters vs Models ===")
    print(f"{'='*70}")
    print(summary.to_string(index=False))

    combined.to_csv(os.path.join(output_path, "irr_vs_avg_human_detail.csv"), index=False)
    summary.to_csv(os.path.join(output_path, "irr_vs_avg_human_summary.csv"), index=False)
    print(f"\n[INFO] Wrote IRR comparison to {output_path}/irr_vs_avg_human_*.csv")


def evaluate_sheets(file_path: str, gold_dfs: Optional[Dict[str, pd.DataFrame]] = None, exclude_ctx: Optional[List[str]] = None, exclude_instr: Optional[List[str]] = None, cot_only: bool = False, avg_human_df: Optional[pd.DataFrame] = None, model_kappa_collector: Optional[list] = None):
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
        # gold_df = gold_dfs[primary_rater]
        gold_df = avg_human_df if avg_human_df is not None else gold_dfs[primary_rater]
        # print(f"[INFO] Using primary gold from CLI: {primary_rater} ({len(gold_df)} sentences)")
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
        KNOWN_PREFIXES = (
            # Model prefixes (full config names like GPT41_ZS_CTX2t_noLeg)
            "GPT", "BERT", "PHI", "MODERNBERT", "MODERN-BERT",
            "GEM", "GEMINI", "QWEN", "QWQ", "LLAMA", "TEST_",
            # Instruction prefixes (stripped names like ZS_CTX2t_noLeg)
            "ZS", "FS", "ZSCOT", "FSCOT",
        )
        if not any(name_up.startswith(p) for p in KNOWN_PREFIXES):
            continue

        df = drop_features_column(df_raw)
        df = normalize_sentence_column(df, sheet_name)
        if df is None:
            continue
        df = df.dropna(subset=["sentence"])
        df = safe_strip_sentence_col(df)

        # Determine canonical name for this config.
        # When model prefix is stripped (sheet = "ZS_CTX2t_noLeg", model_name = "GPT41_ZS_CTX2t_noLeg"),
        # use the SHORT sheet tab as display name so plot labels stay clean.
        # The full name in model_name is used by parse_factors() for model identity.
        if "model_name" in df.columns:
            df = df.drop(columns=["model_name"])
        canonical_name = str(sheet_name)

        if exclude_ctx:
            _factors = parse_factors(canonical_name)
            if _factors.get("ctx") in exclude_ctx:
                print(f"[INFO] Skipping {canonical_name} (ctx={_factors.get('ctx')} excluded by --exclude-ctx)")
                continue

        if exclude_instr or cot_only:
            _factors = parse_factors(canonical_name)
            _instr = _factors.get("instr")
            if cot_only and _instr not in ("ZScot", "FScot"):
                print(f"[INFO] Skipping {canonical_name} (instr={_instr}, not a COT condition; --cot-only active)")
                continue
            if exclude_instr and _instr in exclude_instr:
                print(f"[INFO] Skipping {canonical_name} (instr={_instr} excluded by --exclude-instr)")
                continue

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

    # Apply display-name substitutions (e.g. GEMINI → "Gemini 2.5 Flash") for all plots
    all_eval["model"] = all_eval["model"].apply(prettify_model_label)
    eval_results = [df.assign(model=df["model"].apply(prettify_model_label)) for df in eval_results]

    # Derive model label from workbook filename (e.g. "GPT41_Combined" → "GPT41")
    model_label = prettify_model_label(file_basename.replace("_Combined", "").replace("_combined", ""))

    # Pairwise deltas across all configs in this workbook
    work_eval = all_eval.dropna(subset=["ctx"]).copy() if "ctx" in all_eval.columns else all_eval.copy()
    auto_pairwise_deltas(
        work_eval,
        factors=["feature_set", "ctx", "instr", "leg", "trial"],
        output_base=output_base,
        prefix=f"{model_label}_",
    )

    # ==========================================
    # Significance tests + summary visualizations
    # ==========================================

    sig_df = auto_significance_tests(
        work_eval,
        factors=["feature_set", "ctx", "instr", "leg", "trial"],
        output_base=output_base,
        prefix=f"{model_label}_",
    )

    # Forest plot for variable effects
    if sig_df is not None:
        plot_variable_effects(
            sig_df,
            save_path=os.path.join(output_base, f"{model_label}_variable_effects_forest.png"),
            title=f"{model_label}: Variable Effects on F1 (forest plot)",
        )

    # Pairwise config-impact panel: one subplot per factor, bars per model
    plot_config_impact_panel(
        work_eval,
        output_base=output_base,
        prefix=f"{model_label}_",
        factors=["ctx", "instr", "cot", "leg", "feature_set"],
        model_label=model_label,
    )

    # Context depth breakdown: CTX1t vs CTX5 vs noCTX pairwise
    plot_ctx_breakdown(
        work_eval,
        output_base=output_base,
        prefix=f"{model_label}_",
        model_label=model_label,
    )

    # Config leaderboard
    plot_config_leaderboard(
        all_eval,
        save_path=os.path.join(output_base, f"{model_label}_config_leaderboard.png"),
        title=f"{model_label}: Configuration Leaderboard (ranked by macro-F1)",
    )

    # Summary table CSV
    generate_summary_table(
        all_eval,
        save_path=os.path.join(output_base, f"{model_label}_summary_leaderboard.csv"),
    )

    # Dedicated plots for 25-only
    ext_eval = all_eval[all_eval["feature_set"] == "25"].copy()
    if not ext_eval.empty:
        ext_models = ext_eval["model"].unique().tolist()
        # ext_dfs = [df for df in eval_results if df["model"].iloc[0] in ext_models]
        ext_dfs = [df for df in eval_results 
           if df["model"].iloc[0] in ext_models 
           and all_eval.loc[all_eval["model"] == df["model"].iloc[0], "feature_set"].eq("25").any()]
        plot_model_metrics(
            eval_dfs=ext_dfs,
            metric="f1",
            style="heatmap",
            align="intersection",
            title=f"{model_label} EXTENDED (25): F1 by feature",
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
            title=f"{model_label} MASIS (17): F1 by feature",
            save_path=os.path.join(output_base, "MAS17_F1_heatmap_intersection.png"),
            figsize=(14, 10),
        )

    # Per-model mean/variance of F1 across configs
    mean_f1 = all_eval.groupby("feature")["f1"].mean().sort_values(ascending=False).reset_index(name="mean_f1")
    mean_f1.to_csv(os.path.join(output_base, f"{model_label}_mean_f1_per_feature.csv"), index=False)
    _dissertation_fig2_feature_f1(mean_f1, output_base, model_label)
    _dissertation_fig3_regression(all_eval, output_base, model_label)

    var_f1 = all_eval.groupby("feature")["f1"].var(ddof=1).reset_index(name="var_f1")
    var_f1.to_csv(os.path.join(output_base, f"{model_label}_var_f1_per_feature.csv"), index=False)

    # Cross-config comparisons (use model name from filename)
    all_models_label = model_label

    plot_model_metrics(
        eval_dfs=eval_results,
        metric="f1",
        style="heatmap",
        align="intersection",
        title=f"{model_label}: F1 by AAE feature (heatmap, shared features only)",
        save_path=os.path.join(output_base, f"{all_models_label}_F1_heatmap_intersection.png"),
        figsize=(22, 12),
    )

    plot_overall_f1_scores(
        eval_dfs=eval_results,
        align="intersection",
        save_path=os.path.join(output_base, f"{all_models_label}_Overall_F1_intersection.png"),
        title=f"{model_label}: Overall F1 by Config (shared features only)",
    )

    plot_overall_f1_scores(
        eval_dfs=eval_results,
        align="union",
        save_path=os.path.join(output_base, f"{all_models_label}_Overall_F1_full_union.png"),
        title=f"{model_label}: Overall F1 by Config (all features)",
    )

    plot_f1_scores_per_feature(
        eval_dfs=eval_results,
        align="intersection",
        save_path=os.path.join(output_base, f"{all_models_label}_Per_Feature_F1_intersection.png"),
        title=f"{model_label}: F1 per Feature by Config (shared features only)",
    )

    plot_overall_micro_f1_scores(
        eval_dfs=eval_results,
        align="intersection",
        save_path=os.path.join(output_base, f"{all_models_label}_MicroF1_intersection.png"),
        title=f"{model_label}: Micro-F1 by Config (shared features only)",
    )

    plot_overall_micro_f1_scores(
        eval_dfs=eval_results,
        align="union",
        save_path=os.path.join(output_base, f"{all_models_label}_MicroF1_full_union.png"),
        title=f"{model_label}: Micro-F1 by Config (all features)",
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
            title=f"{model_label}: Errors per Feature by Config",
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

    # Collect model kappas vs average human (if avg_human_df provided)
    if avg_human_df is not None and model_kappa_collector is not None and model_sheets:
        kappa_features = list(EXTENDED_FEATURES) if any(
            f in avg_human_df.columns for f in EXTENDED_FEATURES if f not in MASIS17_FEATURES
        ) else list(MASIS17_FEATURES)
        model_kappas = compute_model_kappas_vs_average(model_sheets, avg_human_df, kappa_features)
        if not model_kappas.empty:
            model_kappas["base_model"] = model_label
            model_kappa_collector.append(model_kappas)


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


def load_rater_sheets(path: str) -> Dict[str, pd.DataFrame]:
    """
    Load each sheet from an Excel file as a separate human rater.
    Sheet name becomes the rater name. Skips sheets with no sentence column.
    """
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    gold_dfs: Dict[str, pd.DataFrame] = {}
    for sheet_name, df in sheets.items():
        if str(sheet_name).startswith("~$"):
            continue
        df = drop_features_column(df)
        df = normalize_sentence_column(df, str(sheet_name))
        if df is None:
            print(f"[WARN] Rater sheet '{sheet_name}' missing sentence column — skipping.")
            continue
        df = df.dropna(subset=["sentence"])
        df = safe_strip_sentence_col(df)
        df = combine_wh_qu(df)
        df = df.drop_duplicates(subset="sentence")
        gold_dfs[str(sheet_name)] = df
        print(f"[INIT] Loaded rater sheet: '{sheet_name}' ({len(df)} sentences)")
    return gold_dfs


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
        "--rater-file", default=None,
        help="Excel file where each sheet is a human rater's labels. "
             "Loads all sheets as raters, computes pairwise IRR, average human labels, "
             "and a model-vs-human comparison report."
    )

    parser.add_argument(
        "--output-dir",
        default="Results/",
        help="Base directory to save evaluation results"
    )

    parser.add_argument(
        "--exclude-ctx",
        nargs="+",
        default=None,
        metavar="CTX_LEVEL",
        help="Context levels to exclude (e.g. --exclude-ctx CTX5). Sheets with these ctx values are skipped.",
    )

    parser.add_argument(
        "--exclude-instr",
        nargs="+",
        default=None,
        metavar="INSTR_LEVEL",
        help="Instruction types to exclude (e.g. --exclude-instr ZScot FScot). "
             "Valid values: ZS, FS, ZScot, FScot.",
    )

    parser.add_argument(
        "--cot-only",
        action="store_true",
        default=False,
        help="Only evaluate COT conditions (ZScot / FScot); skip all non-COT sheets.",
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
    if args.rater_file:
        print(f"[INIT] Loading rater sheets from: {args.rater_file}")
        gold_dfs = load_rater_sheets(args.rater_file)
        print(f"[INIT] Loaded {len(gold_dfs)} rater(s): {list(gold_dfs.keys())}")
    elif args.gold:
        gold_dfs = {}
        for gpath in args.gold:
            rater_name = os.path.splitext(os.path.basename(gpath))[0]
            print(f"[INIT] Loading gold file: {gpath} (rater: {rater_name})")
            gold_dfs[rater_name] = load_gold_file(gpath)
        print(f"[INIT] Loaded {len(gold_dfs)} gold rater(s): {list(gold_dfs.keys())}")

    # Compute average human labels and run IRR if 2+ raters
    avg_human_df = None
    human_kappas = pd.DataFrame()
    if gold_dfs and len(gold_dfs) >= 2:
        irr_features = list(EXTENDED_FEATURES) if any(
            f in list(gold_dfs.values())[0].columns for f in EXTENDED_FEATURES if f not in MASIS17_FEATURES
        ) else list(MASIS17_FEATURES)
        avg_human_df = compute_average_human_labels(gold_dfs, irr_features)
        print(f"[INIT] Computed average human labels ({len(avg_human_df)} sentences, {len(irr_features)} features)")
        human_kappas = compute_interrater_reliability(gold_dfs, irr_features, output_dir, avg_human_df=avg_human_df)

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
    model_kappa_collector: list = []
    for filepath in filepaths:
        try:
            evaluate_sheets(
                filepath,
                gold_dfs=gold_dfs,
                exclude_ctx=args.exclude_ctx,
                exclude_instr=args.exclude_instr,
                cot_only=args.cot_only,
                avg_human_df=avg_human_df,
                model_kappa_collector=model_kappa_collector,
            )
        except Exception as e:
            print(f"\n[ERROR] Failed to evaluate {filepath}: {e}")
            import traceback
            traceback.print_exc()

    # 5. Write model vs human comparison report + Figure 1
    if avg_human_df is not None and (not human_kappas.empty or model_kappa_collector):
        all_model_kappas = pd.concat(model_kappa_collector, ignore_index=True) if model_kappa_collector else pd.DataFrame()
        write_irr_model_comparison(human_kappas, all_model_kappas, output_dir)
        _dissertation_fig1_irr(human_kappas, all_model_kappas, output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Dissertation summary figures (Figures 1, 2, 3)
# ─────────────────────────────────────────────────────────────────────────────

def _dissertation_fig1_irr(human_kappas: pd.DataFrame, all_model_kappas: pd.DataFrame, out_dir: str):
    """
    Figure 1 – IRR comparison: human raters vs LLM model families.
    Shows the task is genuinely hard (humans ≠ perfect) and models
    reach a similar agreement level with the average human annotation.
    """
    if human_kappas.empty and all_model_kappas.empty:
        return

    # ---- human summary: mean kappa per rater ----
    h_sum = (
        human_kappas.groupby("entity")["cohens_kappa"]
        .agg(mean="mean", sem=lambda x: x.sem())
        .reset_index()
    )

    # ---- model summary: mean kappa per base_model ----
    if not all_model_kappas.empty and "base_model" in all_model_kappas.columns:
        m_sum = (
            all_model_kappas.groupby("base_model")["cohens_kappa"]
            .agg(mean="mean", sem=lambda x: x.sem())
            .reset_index()
            .rename(columns={"base_model": "entity"})
        )
    else:
        m_sum = pd.DataFrame(columns=["entity", "mean", "sem"])

    fig, ax = plt.subplots(figsize=(10, 6))

    human_colors = ["#c0392b", "#e67e22", "#8e44ad"]
    model_colors = ["#1a5276", "#6c3483", "#1e8449", "#117a65", "#7d6608", "#884ea0"]

    x = 0
    xticks, xlabels = [], []

    for i, (_, row) in enumerate(h_sum.sort_values("mean", ascending=False).iterrows()):
        ax.bar(x, row["mean"], yerr=row["sem"], color=human_colors[i % len(human_colors)],
               width=0.6, capsize=5, alpha=0.85, edgecolor="white", linewidth=1.2)
        xticks.append(x); xlabels.append(f"{row['entity']}\n(human)")
        x += 1

    x += 0.5  # gap

    for i, (_, row) in enumerate(m_sum.sort_values("mean", ascending=False).iterrows()):
        ax.bar(x, row["mean"], yerr=row["sem"], color=model_colors[i % len(model_colors)],
               width=0.6, capsize=5, alpha=0.85, edgecolor="white", linewidth=1.2)
        xticks.append(x); xlabels.append(f"{row['entity']}\n(model)")
        x += 1

    if not h_sum.empty:
        avg_h = h_sum["mean"].mean()
        ax.axhline(avg_h, ls="--", color="gray", lw=1.2, alpha=0.7,
                   label=f"Avg human κ = {avg_h:.2f}")

    sep = len(h_sum) - 0.5 + 0.25
    ax.axvline(sep, color="lightgrey", ls=":", lw=1.5)

    n_h = len(h_sum)
    if n_h:
        ax.text((n_h - 1) / 2, 0.97, "Human raters", ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=9, color="gray")
    n_m = len(m_sum)
    if n_m:
        ax.text(n_h + 0.5 + (n_m - 1) / 2, 0.97, "LLM models", ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=9, color="gray")

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("Mean Cohen's κ vs. Average Human Annotation", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Inter-Rater Agreement: Human Annotators vs. LLM Configurations\n"
        "(across AAE linguistic features; error bars = ±1 SEM)",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(out_dir, "fig1_irr_human_vs_model.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[FIG1] Saved {out}")


def _dissertation_fig2_feature_f1(mean_f1: pd.DataFrame, out_dir: str, model_label: str):
    """
    Figure 2 – Feature-level F1 bar chart for one model family.
    Coloured by performance tier; saved to the model's output folder.
    """
    if mean_f1.empty:
        return

    df = mean_f1.sort_values("mean_f1", ascending=True).reset_index(drop=True)

    def _color(f):
        if pd.isna(f): return "#bdc3c7"
        return "#27ae60" if f >= 0.7 else ("#e67e22" if f >= 0.3 else "#e74c3c")

    bar_colors = [_color(v) for v in df["mean_f1"]]

    fig, ax = plt.subplots(figsize=(12, max(7, len(df) * 0.38)))
    ax.barh(df["feature"], df["mean_f1"].fillna(0),
            color=bar_colors, height=0.65, edgecolor="white", linewidth=0.8, alpha=0.85)

    ax.axvline(0.7, color="#27ae60", ls="--", lw=1, alpha=0.6)
    ax.axvline(0.3, color="#e74c3c", ls="--", lw=1, alpha=0.6)

    import matplotlib.patches as _mp
    legend_patches = [
        _mp.Patch(color="#27ae60", label="Good  (F1 ≥ 0.7)"),
        _mp.Patch(color="#e67e22", label="Moderate  (0.3 – 0.7)"),
        _mp.Patch(color="#e74c3c", label="Poor  (F1 < 0.3)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    ax.set_xlabel("Mean F1 (averaged across all prompt configurations)", fontsize=11)
    ax.set_title(
        f"{model_label}: Performance by AAE Linguistic Feature\n"
        "(mean F1 across all configs)",
        fontsize=12,
    )
    ax.set_xlim(-0.02, 1.1)
    plt.tight_layout()
    out = os.path.join(out_dir, f"{model_label}_fig2_feature_f1.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[FIG2] Saved {out}")


def _dissertation_fig3_regression(all_eval: pd.DataFrame, out_dir: str, model_label: str):
    """
    Figure 3 – OLS regression: macro_f1 ~ instr + ctx + leg (within one model family).
    Shows which prompt configuration parameters actually matter.
    """
    if not _HAS_STATSMODELS:
        print("[FIG3] statsmodels not installed — skipping regression figure.")
        return

    needed = {"model", "feature", "f1", "instr", "ctx", "leg"}
    if not needed.issubset(all_eval.columns):
        print(f"[FIG3] Missing columns for regression: {needed - set(all_eval.columns)}")
        return

    # macro_f1 per config = mean F1 across features
    lb = (
        all_eval.dropna(subset=["instr", "ctx", "leg"])
        .groupby(["model", "instr", "ctx", "leg"])["f1"]
        .mean()
        .reset_index()
        .rename(columns={"f1": "macro_f1"})
    )
    lb["ctx"] = lb["ctx"].fillna("noCTX")

    if lb.empty or lb["macro_f1"].isna().all():
        return

    # pick reference levels: most common or sensible defaults
    ref_instr = "ZS" if "ZS" in lb["instr"].values else lb["instr"].iloc[0]
    ref_ctx   = "noCTX" if "noCTX" in lb["ctx"].values else lb["ctx"].iloc[0]
    ref_leg   = "noLeg" if "noLeg" in lb["leg"].values else lb["leg"].iloc[0]

    formula = (
        f"macro_f1 ~ "
        f"C(instr, Treatment('{ref_instr}')) + "
        f"C(ctx, Treatment('{ref_ctx}')) + "
        f"C(leg, Treatment('{ref_leg}'))"
    )
    try:
        result = _smf.ols(formula, data=lb).fit()
    except Exception as e:
        print(f"[FIG3] Regression failed: {e}")
        return

    # Save table
    tbl = pd.DataFrame({
        "predictor": result.params.index,
        "coef":      result.params.values,
        "se":        result.bse.values,
        "t":         result.tvalues.values,
        "p":         result.pvalues.values,
        "ci_low":    result.conf_int()[0].values,
        "ci_high":   result.conf_int()[1].values,
    })
    tbl_path = os.path.join(out_dir, f"{model_label}_fig3_regression.csv")
    tbl.to_csv(tbl_path, index=False)

    # Forest plot (drop intercept)
    coef  = result.params.drop("Intercept")
    ci    = result.conf_int().drop("Intercept")
    pvals = result.pvalues.drop("Intercept")

    def _group_color(k):
        if "instr" in k: return "#8e44ad"
        if "ctx"   in k: return "#16a085"
        return "#e67e22"

    import matplotlib.patches as _mp
    dot_colors = [_group_color(k) for k in coef.index]
    sig        = pvals < 0.05

    # Clean up labels
    def _clean(k):
        k = re.sub(r"C\(\w+, Treatment\('[^']+'\)\)\[T\.([^\]]+)\]", r"\1", k)
        return k

    labels = [_clean(k) for k in coef.index]

    fig, ax = plt.subplots(figsize=(9, max(5, len(coef) * 0.55)))
    y_pos = np.arange(len(coef))

    for i, (c, lo, hi, col, is_sig) in enumerate(zip(
        coef.values,
        coef.values - ci[0].values,
        ci[1].values - coef.values,
        dot_colors, sig
    )):
        ax.plot([c - lo, c + hi], [i, i], color=col,
                lw=2.5 if is_sig else 1.2, alpha=0.9 if is_sig else 0.5)
        ax.scatter(c, i, color=col, s=120 if is_sig else 55, zorder=5,
                   edgecolors="white", linewidth=1,
                   marker="D" if is_sig else "o")

    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Regression Coefficient (effect on macro F1)", fontsize=11)
    ax.set_title(
        f"{model_label}: Which Config Parameters Predict Performance?\n"
        f"OLS  |  n={len(lb)}  |  R²={result.rsquared:.3f}  |  Adj. R²={result.rsquared_adj:.3f}",
        fontsize=12,
    )

    group_patches = [
        _mp.Patch(color="#8e44ad", label="Instruction style"),
        _mp.Patch(color="#16a085", label="Context window"),
        _mp.Patch(color="#e67e22", label="Legitimacy frame"),
        plt.Line2D([0],[0], marker="D", color="w", markerfacecolor="gray",
                   markersize=8, label="p < 0.05"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="gray",
                   markersize=6, label="n.s."),
    ]
    ax.legend(handles=group_patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    out = os.path.join(out_dir, f"{model_label}_fig3_regression.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[FIG3] Saved {out}")


if __name__ == "__main__":
    main()
