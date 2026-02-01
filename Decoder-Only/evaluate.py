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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from multi_prompt_configs import (
    EXTENDED_FEATURES,
    MASIS_FEATURES,
)

"""
nlprun -q jag -p standard -r 40G -c 2 -t 4:00:00 \
  -n eval-gpt-sheets \
  -o slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation/Decoder-Only/GPT && \
   mkdir -p slurm_logs && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python evaluate.py"
"""

# Treat wh-qu as part of the MASIS/17 set (it may be derived from wh-qu1/wh-qu2)
MASIS17_FEATURES = list(MASIS_FEATURES)
if "wh-qu" not in MASIS17_FEATURES:
    MASIS17_FEATURES.append("wh-qu")

# enforce consistent delta direction (b - a)
PREFERRED_LEVEL_ORDER = {
    "ctx": ("noCTX", "CTX"),          # CTX - noCTX
    "instr": ("ZS", "FSCOT"),         # FSCOT - ZS
    "feature_set": ("17", "25"),      # 25 - 17
    "trial": ("A", "B"),              # B - A
}

# one definition only
feat_thresholds = {
    "multiple-neg": 0.5,
}

output_dir = "data/results"
os.makedirs(output_dir, exist_ok=True)


def drop_features_column(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["FEATURES", "Source"], errors="ignore")


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
      GPT_17_ZS_CTX_A
      GEMINI_25_FSCOT_noCTX_B
    Returns dict with keys:
      model in {"GPT", "GEMINI"} or None
      feature_set in {"17","25"} or None
      instr in {"ZS","FSCOT"} or None
      ctx in {"CTX","noCTX"} or None
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
    if "FSCOT" in toks or "_FSCOT_" in f"_{m}_":
        instr = "FSCOT"
    elif "ZS" in toks or "_ZS_" in f"_{m}_":
        instr = "ZS"

    ctx = None
    if "NOCTX" in toks or "_NOCTX" in m:
        ctx = "noCTX"
    elif "CTX" in toks or "_CTX" in m:
        ctx = "CTX"

    trial = None
    if toks and toks[-1] in ("A", "B"):
        trial = toks[-1]

    # Add 'model' to the returned dictionary
    return {"feature_set": feature_set, "instr": instr, "ctx": ctx, "trial": trial}

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
        print(f"[INFO] No non-empty features for {model_name}; skipping confusion-matrix plot.")
        # Return an empty DataFrame so caller can handle it
        return pd.DataFrame()
            
    plot_per_feature_confusion_matrix(
        cm_df,
        model_name,
        filtered_features,
        save_path=os.path.join(output_base, f"{model_name}_confusion_matrix.png"),
    )

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


def evaluate_sheets(file_path: str):
    print(f"\n===============================")
    print(f"=== Evaluating {file_path} ===")
    print(f"===============================\n")

    file_basename = os.path.splitext(os.path.basename(file_path))[0]
    output_base = os.path.join(output_dir, file_basename)
    os.makedirs(output_base, exist_ok=True)

    sheets = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")

    gold_df = sheets.get("Gold")
    if gold_df is None:
        print(f"[WARN] No 'Gold' sheet in {file_path}. Skipping.")
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

    for sheet_name, df_raw in sheets.items():
        if sheet_name == "Gold":
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
        if not (name_up.startswith("GPT") or name_up.startswith("BERT") or name_up.startswith("PHI") or name_up.startswith("MODERNBERT") or name_up.startswith("GEM") or name_up.startswith("GEMINI")):
            continue

        df = drop_features_column(df_raw)
        df = normalize_sentence_column(df, sheet_name)
        if df is None:
            continue
        df = df.dropna(subset=["sentence"])
        df = safe_strip_sentence_col(df)

        df = combine_wh_qu(df)
        df = df.drop_duplicates(subset="sentence")

        model_sheets[sheet_name] = df

    if not model_sheets:
        print(f"[WARN] No model sheets (GPT_*/BERT/PHI/MODERNBERT) found in {file_path}.")
        return

    # [EXISTING CODE] ... 
    # modelsheets[sheetname] = df 
    # if not modelsheets:
    #     print(f"WARN: No model sheets found...")
    #     return
    
    # --- INSERT START: STRICT INTERSECTION FILTERING ---
    # 1. Start with Gold sentences
    common_ids = set(gold_df['sentence'].dropna())
    initial_gold_count = len(common_ids)
    print(f"DEBUG: Starting with {initial_gold_count} Gold sentences.")

    # 2. Intersect with EVERY model sheet found
    for name, df in model_sheets.items():
        model_sents = set(df['sentence'].dropna())
        before_count = len(common_ids)
        common_ids = common_ids.intersection(model_sents)
        dropped = before_count - len(common_ids)
        if dropped > 0:
            print(f"DEBUG: Dropping {dropped} sentences missing from model '{name}' (Retaining {len(common_ids)})")

    # 3. Safety Check
    if not common_ids:
        print("CRITICAL ERROR: Intersection of sentence IDs is EMPTY. No sentences are shared by all models.")
        return # Or handle gracefully depending on preference
    
    if len(common_ids) < initial_gold_count:
        print(f"STRICT MODE: Pruned evaluation set from {initial_gold_count} to {len(common_ids)} shared sentences.")

    # 4. Apply Filter to Gold
    gold_df = gold_df[gold_df['sentence'].isin(common_ids)].copy()

    # 5. Apply Filter to All Models
    for name in model_sheets:
        model_sheets[name] = model_sheets[name][model_sheets[name]['sentence'].isin(common_ids)].copy()
    # --- INSERT END ---


    eval_results: list[pd.DataFrame] = []
    per_model_errors: dict[str, pd.DataFrame] = []
    per_model_errors = {}

    # Keep per-model detected feature-set num to ensure factors are filled later
    model_feature_set_num: Dict[str, str] = {}

    for sheet_name, model_df in model_sheets.items():
        detected = detect_feature_set(sheet_name, model_df)
        if detected is None:
            print(f"[SKIP] {sheet_name}: could not determine feature set.")
            continue

        features = MASIS17_FEATURES if detected == "masis" else EXTENDED_FEATURES
        model_feature_set_num[sheet_name] = ("17" if detected == "masis" else "25")

        model_df = binarize_if_probabilistic(model_df, features, str(sheet_name))
        model_sheets[sheet_name] = model_df

        print(f"\n--- Evaluating model sheet: {sheet_name} (feature set: {detected}, {len(features)} features) ---")

        model_eval = evaluate_model(
            model_df=model_df,
            truth_df=gold_df,
            model_name=str(sheet_name),
            features=features,
            output_base=output_base,
        )

        if not model_eval.empty:
            eval_results.append(model_eval)

            err_df = build_error_df(
                model_df=model_df,
                gold_df=gold_df,
                features=features,
                model_name=str(sheet_name),
                rationale_df=rationale_sheets.get(sheet_name),
            )
            per_model_errors[str(sheet_name)] = err_df

            rat_df = rationale_sheets.get(sheet_name)
            if rat_df is not None:
                try:
                    annotated = build_annotated_rationales(
                        pred_df=model_df,
                        rationale_df=rat_df,
                        truth_df=gold_df,
                        features=features,
                        only_disagreements=True,
                        max_rows=None,
                    )
                    out_csv = os.path.join(output_base, f"{sheet_name}_annotated_rationales.csv")
                    annotated.to_csv(out_csv, index=False)
                    print(f"[INFO] Wrote annotated rationales for {sheet_name} to {out_csv}")
                except Exception as e:
                    print(f"[WARN] Could not build annotated rationales for {sheet_name}: {e}")

    eval_results = [df for df in eval_results if not df.empty]
    if not eval_results:
        print("[WARN] No non-empty evaluation results.")
        return

    all_eval = pd.concat(eval_results, ignore_index=True)

    # Trial-specific plots (if trial parsed)
    for trial in ["A", "B"]:
        eval_dfs_trial = [df for df in eval_results if parse_factors(df["model"].iloc[0]).get("trial") == trial]
        if not eval_dfs_trial:
            continue

        label = f"TRIAL_{trial}"

        plot_model_metrics(
            eval_dfs=eval_dfs_trial,
            metric="f1",
            style="heatmap",
            align="intersection",
            title=f"{label}: F1 by feature (shared features only)",
            save_path=os.path.join(output_base, f"{label}_F1_heatmap_intersection.png"),
            figsize=(14, 10),
        )

        plot_overall_f1_scores(
            eval_dfs=eval_dfs_trial,
            align="intersection",
            save_path=os.path.join(output_base, f"{label}_Overall_F1_intersection.png"),
        )

        plot_overall_micro_f1_scores(
            eval_dfs=eval_dfs_trial,
            align="intersection",
            save_path=os.path.join(output_base, f"{label}_MicroF1_intersection.png"),
        )

        plot_f1_scores_per_feature(
            eval_dfs=eval_dfs_trial,
            align="intersection",
            save_path=os.path.join(output_base, f"{label}_Per_Feature_F1_intersection.png"),
        )

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
        factors=["feature_set", "ctx", "instr", "trial"],
        output_base=output_base,
        prefix="GPT_",
    )

    # all-model deltas (only where feature_set and ctx exist)
    auto_pairwise_deltas(
        all_eval.dropna(subset=["feature_set", "ctx"]),
        factors=["feature_set", "ctx"],
        output_base=output_base,
        prefix="ALLMODELS_",
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
        plot_model_metrics(
            eval_dfs=ext_dfs,
            metric="f1",
            style="bar",
            align="intersection",
            title="EXTENDED (25): F1 by feature (shared across 25-feature models)",
            save_path=os.path.join(output_base, "EXT25_F1_bar_intersection.png"),
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
        plot_model_metrics(
            eval_dfs=mas_dfs,
            metric="f1",
            style="bar",
            align="intersection",
            title="MASIS (17): F1 by feature (shared across 17-feature models)",
            save_path=os.path.join(output_base, "MAS17_F1_bar_intersection.png"),
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
        style="bar",
        align="intersection",
        title="Model F1 by AAE feature (shared features only)",
        save_path=os.path.join(output_base, f"{all_models_label}_F1_bar_intersection.png"),
    )

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
        "--output-dir", 
        default="data/results", 
        help="Base directory to save evaluation results (default: data/results)"
    )

    args = parser.parse_args()

    # 1. Update the global output directory variable used by evaluate_sheets
    global output_dir
    output_dir = args.output_dir
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        print(f"[INFO] Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    # 2. Resolve input paths (handle files, directories, and glob patterns)
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

    # 3. Run evaluation for each file
    for filepath in filepaths:
        try:
            evaluate_sheets(filepath)
        except Exception as e:
            print(f"\n[ERROR] Failed to evaluate {filepath}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
