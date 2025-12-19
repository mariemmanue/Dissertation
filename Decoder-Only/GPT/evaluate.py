import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import List, Optional, Literal, Sequence
from gpt_experiments import (
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

feat_thresholds = {
    "multiple-neg": 0.5,  # Example threshold
}

output_dir = "data/results"
os.makedirs(output_dir, exist_ok=True)

def drop_features_column(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=['FEATURES', 'Source'], errors='ignore')


def combine_wh_qu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines wh-qu1/wh-qu2 (or wh_qu1/wh_qu2) into a single wh-qu.
    If wh-qu already exists, do nothing.
    """
    out = df.copy()

    if "wh-qu" in out.columns:
        return out

    a = "wh-qu1" if "wh-qu1" in out.columns else ("wh_qu1" if "wh_qu1" in out.columns else None)
    b = "wh-qu2" if "wh-qu2" in out.columns else ("wh_qu2" if "wh_qu2" in out.columns else None)

    if a and b:
        out["wh-qu"] = out[[a, b]].max(axis=1)
        out = out.drop(columns=[a, b])

    return out

def parse_factors(sheet_name: str) -> dict:
    """
    Parse condition factors out of sheet names like:
      GPT_17_ZS_CTX_A
      GPT_25_FSCOT_noCTX_B
    Returns dict with keys:
      feature_set in {"17","25"} or None
      instr in {"ZS","FSCOT"} or None
      ctx in {"CTX","noCTX"} or None
      trial in {"A","B"} or None
    """
    m = sheet_name.upper()

    # split tokens, keep order
    toks = [t for t in m.split("_") if t]

    # feature set
    feature_set = None
    if "17" in toks or "_17_" in f"_{m}_":
        feature_set = "17"
    elif "25" in toks or "_25_" in f"_{m}_":
        feature_set = "25"

    # instruction
    instr = None
    if "FSCOT" in toks or "_FSCOT_" in f"_{m}_":
        instr = "FSCOT"
    elif "ZS" in toks or "_ZS_" in f"_{m}_":
        instr = "ZS"

    # context (prefer explicit noCTX)
    ctx = None
    if "NOCTX" in toks or "_NOCTX" in m:
        ctx = "noCTX"
    elif "CTX" in toks or "_CTX" in m:
        ctx = "CTX"

    # trial A/B at end
    trial = None
    if toks and toks[-1] in ("A", "B"):
        trial = toks[-1]

    return {"feature_set": feature_set, "instr": instr, "ctx": ctx, "trial": trial}


def parse_trial(model: str):
    m = model.upper()
    if m.endswith("_A"):
        return "A"
    if m.endswith("_B"):
        return "B"
    return None


def build_annotated_rationales(pred_df, rationale_df, truth_df, features, only_disagreements=True, max_rows=None):
    pred_df = pred_df.copy()
    rationale_df = rationale_df.copy()
    truth_df = truth_df.copy()

    pred_df["sentence"] = pred_df["sentence"].str.strip()
    rationale_df["sentence"] = rationale_df["sentence"].str.strip()
    truth_df["sentence"] = truth_df["sentence"].str.strip()

    merged = truth_df.merge(pred_df, on="sentence", suffixes=("_gold", "_pred"))
    merged = merged.merge(rationale_df, on="sentence", how="left", suffixes=("", "_rat"))

    rows = []
    for _, row in merged.iterrows():
        sent = row["sentence"]
        for feat in features:
            gold_col = f"{feat}_gold"
            pred_col = f"{feat}_pred"
            rat_col = feat

            if gold_col not in row or pred_col not in row:
                continue

            gold_v = row.get(gold_col, None)
            pred_v = row.get(pred_col, None)

            # Skip row if any value is NaN
            if pd.isna(gold_v) or pd.isna(pred_v):
                continue

            gold_v = int(gold_v)
            pred_v = int(pred_v)

            rationale = row.get(rat_col, "")
            if pd.isna(rationale):
                rationale = ""

            if only_disagreements and gold_v == pred_v:
                continue

            rows.append({
                "sentence": sent,
                "feature": feat,
                "gold": gold_v,
                "pred": pred_v,
                "model_rationale": rationale,
            })

    out_df = pd.DataFrame(rows)
    if max_rows is not None:
        out_df = out_df.head(max_rows)
    return out_df

def plot_overall_micro_f1_scores(
    eval_dfs: List[pd.DataFrame],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
    align: Literal["union", "intersection"] = "intersection",
):
    # Optionally restrict to shared features only
    if align == "intersection":
        shared = None
        for df in eval_dfs:
            feats = set(df["feature"].unique())
            shared = feats if shared is None else shared.intersection(feats)
        shared = shared or set()
        filtered = []
        for df in eval_dfs:
            filtered.append(df[df["feature"].isin(shared)].copy())
        eval_dfs = filtered
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

    out = pd.DataFrame.from_dict(micro_scores, orient="index", columns=["micro_F1"])
    out = out.sort_values("micro_F1", ascending=False)

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
    plt.show()



def plot_overall_f1_scores(eval_dfs: List[pd.DataFrame], save_path: Optional[str] = None, figsize: tuple = (10, 8), align: Literal["union", "intersection"] = "intersection"):
    """
    Calculate overall F1 scores, optionally only on shared features.
    
    Args:
        eval_dfs: List of evaluation DataFrames with columns ['model', 'feature', 'f1']
        save_path: Optional path to save the plot
        figsize: Figure size tuple
        align: "intersection" to only use features present in all models, "union" to use all features
    """
    overall_f1_scores = {}
    
    if align == "intersection":
        # Find features present in all models
        all_features = set()
        for df in eval_dfs:
            if not df.empty:
                all_features.update(df['feature'].unique())
        
        shared_features = all_features.copy()
        for df in eval_dfs:
            if not df.empty:
                df_features = set(df['feature'].unique())
                shared_features = shared_features.intersection(df_features)
        
        # Filter each df to only shared features
        filtered_dfs = []
        for df in eval_dfs:
            if not df.empty:
                filtered_df = df[df['feature'].isin(shared_features)].copy()
                filtered_dfs.append(filtered_df)
            else:
                filtered_dfs.append(df)
        eval_dfs = filtered_dfs
        print(f"Using {len(shared_features)} shared features for overall F1: {sorted(shared_features)}")
    
    for df in eval_dfs:
        if not df.empty:
            model_name = df['model'].iloc[0]
            overall_f1 = df['f1'].mean()
            overall_f1_scores[model_name] = overall_f1

    overall_f1_df = pd.DataFrame.from_dict(overall_f1_scores, orient='index', columns=['F1 Score'])
    overall_f1_df = overall_f1_df.sort_values(by='F1 Score', ascending=False)

    ax = overall_f1_df.plot(kind='bar', figsize=figsize, legend=False)
    ax.set_ylabel('F1 Score')
    ax.set_title(
    "Overall F1 Scores by Model (shared features only)"
    if align == "intersection"
    else "Overall F1 Scores by Model (each model’s evaluated features; not strictly comparable)"
    )
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_f1_scores_per_feature(eval_dfs: List[pd.DataFrame], save_path: Optional[str] = None, figsize: tuple = (14, 10), align: Literal["union", "intersection"] = "intersection"):
    """
    Plot F1 scores per feature, optionally only for shared features.
    
    Args:
        eval_dfs: List of evaluation DataFrames
        save_path: Optional path to save the plot
        figsize: Figure size tuple
        align: "intersection" to only show features present in all models, "union" to show all features
    """
    all_eval = pd.concat(eval_dfs, ignore_index=True)
    pivot = all_eval.pivot(index="feature", columns="model", values="f1")
    
    if align == "intersection":
        # Only keep features present in all models
        present_counts = pivot.notna().sum(axis=1)
        pivot = pivot[present_counts == pivot.shape[1]]
        print(f"Showing {len(pivot)} shared features in heatmap")
    
    pivot = pivot.fillna(0.0).sort_index()

    plt.figure(figsize=figsize)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "F1 score"}
    )
    title = 'F1 Scores per Feature by Model (Shared Features Only)' if align == "intersection" else 'F1 Scores per Feature by Model'
    plt.title(title)
    plt.xlabel('Model')
    plt.ylabel('Feature')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_per_feature_confusion_matrix(cm_data, model_name, features, save_path=None):
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', xticklabels=["TP", "FP", "FN", "TN"], yticklabels=features)
    plt.title(f'Confusion Matrix Per Feature for {model_name}')
    plt.xlabel('Metric')
    plt.ylabel('Feature')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model_df, truth_df, model_name, features, output_base):
    print(f"\n=== {model_name} Evaluation ===")

    if model_df is None:
        print(f"{model_name} data not available. Skipping evaluation.")
        return pd.DataFrame()

    model_df = model_df.drop_duplicates(subset='sentence')
    truth_df = truth_df.drop_duplicates(subset='sentence')

    model_df = combine_wh_qu(model_df)
    truth_df = combine_wh_qu(truth_df)

    available_features = [feat for feat in features if feat in model_df.columns and feat in truth_df.columns]
    shared_columns = ['sentence'] + available_features

    model_df = model_df[shared_columns]
    truth_df = truth_df[shared_columns]

    summary = {
        'model': [],
        'feature': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'TP': [],
        'FP': [],
        'FN': [],
        'TN': []
    }
    skipped_features = []
    cm_data = []

    for feat in available_features:
        merged = pd.merge(
            truth_df[['sentence', feat]],
            model_df[['sentence', feat]],
            on='sentence',
            how='inner'
        ).rename(columns={f"{feat}_x": "y_true", f"{feat}_y": "y_pred"})

        merged = merged.dropna(subset=['y_true', 'y_pred'])

        if merged.empty:
            print(f"Skipping {feat} — no data after dropping NaNs.")
            skipped_features.append(feat)
            continue

        y_true = merged['y_true'].astype(int)
        y_pred = merged['y_pred'].astype(int)

        if y_true.sum() == 0 and y_pred.sum() == 0:
            print(f"Skipping {feat} — no positive instances in ground truth and predictions.")
            skipped_features.append(feat)
            continue

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        TP = int(cm[0, 0])
        FN = int(cm[0, 1])
        FP = int(cm[1, 0])
        TN = int(cm[1, 1])

        summary['model'].append(model_name)
        summary['feature'].append(feat)
        summary['accuracy'].append(acc)
        summary['precision'].append(prec)
        summary['recall'].append(rec)
        summary['f1'].append(f1)
        summary['TP'].append(TP)
        summary['FP'].append(FP)
        summary['FN'].append(FN)
        summary['TN'].append(TN)

        cm_data.append([TP, FP, FN, TN])

    # Filter out skipped features to match cm_data length
    filtered_features = [feat for feat in available_features if feat not in skipped_features]

    # Convert to DataFrame and plot confusion matrix for per-feature counts
    cm_df = pd.DataFrame(cm_data, columns=["TP", "FP", "FN", "TN"], index=filtered_features)
    plot_per_feature_confusion_matrix(cm_df, model_name, filtered_features, save_path=os.path.join(output_base, f'{model_name}_confusion_matrix.png'))

    results = pd.DataFrame(summary)
    print("\n=== Summary Metrics ===")
    print(results.round(4))

    if not results.empty:
        print("\n=== Macro Averages ===")
        print(results[['accuracy', 'precision', 'recall', 'f1']].mean().round(4))
        print("\n=== Confusion Matrix Counts ===")
        print(results[['feature', 'TP', 'FP', 'FN', 'TN']].to_string(index=False))

    print(f"\n=== Skipped {skipped_features} feature(s) with no positives in ground truth or predictions ===")
    return results


def plot_model_metrics(
    *,
    eval_dfs: Optional[List[pd.DataFrame]] = None,
    err_counts: Optional[pd.DataFrame] = None,
    metric: Literal["f1", "errors"] = "f1",
    style: Literal["bar", "heatmap"] = "bar",
    title: str = None,
    figsize: tuple = (12, 6),
    rotation: int = 45,
    save_path: Optional[str] = None,
    align: Literal["union", "intersection"] = "union",
    annotate_heatmap: bool = True
) -> None:

    def _prepare_f1_pivot(dfs: List[pd.DataFrame]) -> pd.DataFrame:
        all_eval = pd.concat(dfs, ignore_index=True)
        pivot = all_eval.pivot(index="feature", columns="model", values="f1")

        if align == "intersection":
            present_counts = pivot.notna().sum(axis=1)
            pivot = pivot[present_counts == pivot.shape[1]]

        pivot = pivot.fillna(0.0).sort_index()
        return pivot

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
        ylab = "AAE feature"
        xlab = "Model"
        fmt = ".2f"
        cmap = "RdYlGn"
    else:
        if err_counts is None:
            raise ValueError("metric='errors' requires err_counts=DataFrame")
        pivot = _prepare_err_pivot(err_counts)
        default_title = "Errors per feature by model"
        vmin, vmax = None, None
        cbar_label = "Number of errors"
        ylab = "Feature"
        xlab = "Model"
        fmt = ".0f"
        cmap = "Reds"

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
        plt.show()

    elif style == "heatmap":
        plt.figure(figsize=figsize)
        sns.heatmap(
            pivot,
            annot=annotate_heatmap,
            fmt=fmt,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar_kws={"label": cbar_label}
        )
        plt.title(title)
        plt.ylabel(ylab)
        plt.xlabel(xlab)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
    else:
        raise ValueError("style must be 'bar' or 'heatmap'")

def build_error_df(
    model_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    features: list[str],
    model_name: str,
    rationale_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    model_sub = model_df.copy()
    gold_sub = gold_df.copy()

    model_sub = combine_wh_qu(model_sub)
    gold_sub = combine_wh_qu(gold_sub)

    needed_cols = ["sentence"] + [f for f in features if f in model_sub.columns and f in gold_sub.columns]
    model_sub = model_sub[needed_cols].copy()
    gold_sub = gold_sub[needed_cols].copy()

    merged = gold_sub.merge(model_sub, on="sentence", suffixes=("_gold", "_pred"))

    # Build a fast lookup: (sentence, feature) -> rationale text
    rat_lookup = {}
    if rationale_df is not None and "sentence" in rationale_df.columns:
        r = rationale_df.copy()
        r["sentence"] = r["sentence"].astype(str).str.strip()
        for feat in features:
            if feat in r.columns:
                # If rationale cells are NaN, treat as ""
                rat_lookup.update({
                    (sent, feat): (txt if pd.notna(txt) else "")
                    for sent, txt in zip(r["sentence"].tolist(), r[feat].tolist())
                })

    rows = []
    for _, row in merged.iterrows():
        sent = row["sentence"]
        for feat in features:
            gold_col = f"{feat}_gold"
            pred_col = f"{feat}_pred"
            if gold_col not in row or pred_col not in row:
                continue

            gv = row[gold_col]
            pv = row[pred_col]
            if pd.isna(gv) or pd.isna(pv):
                continue

            gv = int(gv); pv = int(pv)
            if gv != pv:
                rows.append({
                    "model": model_name,
                    "sentence": sent,
                    "feature": feat,
                    "gold": gv,
                    "pred": pv,
                    "model_rationale": rat_lookup.get((str(sent).strip(), feat), ""),
                })

    return pd.DataFrame(rows)


def auto_pairwise_deltas(
    df: pd.DataFrame,
    *,
    factors: list[str],
    output_base: str,
    prefix: str,
    metrics=("precision", "recall", "f1"),
):
    work = df.copy()
    work = work.dropna(subset=["feature"])

    # keep only factor columns that actually exist
    factors = [f for f in factors if f in work.columns]

    for fac in factors:
        levels = sorted(work[fac].dropna().unique().tolist())
        if len(levels) != 2:
            continue
        a_level, b_level = levels[0], levels[1]

        # robust scalar check even if duplicate columns make work[[f]] multi-col
        other_factors = [
            f for f in factors
            if f != fac and work[[f]].notna().to_numpy().any()
        ]

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

        out_csv = os.path.join(output_base, f"{prefix}delta_{fac}_{a_level}_to_{b_level}_per_feature.csv")
        deltas.to_csv(out_csv, index=False)
        print(f"[INFO] Wrote {out_csv} ({len(deltas)} rows)")








def safe_sheet_name(name: str, max_len: int = 31) -> str:
    """
    Excel sheet names must be <=31 chars and avoid some characters.
    This helper truncates and cleans model names for use as sheet names.
    """
    bad_chars = ['[', ']', '*', '?', '/', '\\', ':']
    for ch in bad_chars:
        name = name.replace(ch, "_")
    return name[:max_len]


def detect_feature_set(sheet_name: str, df: pd.DataFrame) -> Optional[str]:
    sheet_up = sheet_name.upper()
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


# keep your dict name, but use it consistently
feat_thresholds = {"multiple-neg": 0.5}

def binarize_if_probabilistic(df: pd.DataFrame, features: list[str], model_name: str):
    out = df.copy()
    for feat in features:
        if feat not in out.columns:
            continue
        s = out[feat]

        uniq = set(pd.Series(s.dropna()).unique().tolist())
        if uniq.issubset({0, 1, 0.0, 1.0, True, False}):
            out[feat] = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
            continue

        vals = pd.to_numeric(s, errors="coerce")
        if vals.notna().sum() == 0:
            continue  # don't clobber

        thr = feat_thresholds.get(feat, 0.5)
        out[feat] = (vals >= thr).astype(int)

    return out


METRICS_DEFAULT = ("precision", "recall", "f1")

def safe_pivot_get(pivot: pd.DataFrame, metric: str, level: str) -> pd.Series:
    """Return pivot[(metric, level)] if it exists, else NaNs aligned to pivot.index."""
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
    require_metric: str = "f1",     # which metric defines “paired / even”
    aggfunc: str = "first",
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build pivot over `factor_col` and compute (b_level - a_level) deltas for each metric.
    Filters to rows where both levels exist (paired) using `require_metric`.
    """
    # pivot: index_cols x factor_col -> metrics
    pivot = df.pivot_table(
        index=list(index_cols),
        columns=factor_col,
        values=list(metrics),
        aggfunc=aggfunc
    )

    # evenness / pairing filter
    have_both = safe_pivot_get(pivot, require_metric, a_level).notna() & safe_pivot_get(pivot, require_metric, b_level).notna()
    pivot = pivot[have_both]

    out = {}
    for m in metrics:
        d = safe_pivot_get(pivot, m, b_level) - safe_pivot_get(pivot, m, a_level)
        colname = f"d_{m}_{b_level}_minus_{a_level}"
        if prefix:
            colname = f"{prefix}{colname}"
        out[colname] = d

    deltas = pd.DataFrame(out).reset_index()
    return deltas



def evaluate_sheets(file_path: str):
    print(f"\n===============================")
    print(f"=== Evaluating {file_path} ===")
    print(f"===============================\n")

    # Base directory to write results for this workbook
    file_basename = os.path.splitext(os.path.basename(file_path))[0]
    output_base = os.path.join(output_dir, file_basename)
    os.makedirs(output_base, exist_ok=True)

    # Read all sheets (skip Excel temp/lock files later)
    sheets = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")

    # ---- 1. Load Gold sheet ----
    gold_df = sheets.get("Gold")
    if gold_df is None:
        print(f"[WARN] No 'Gold' sheet in {file_path}. Skipping.")
        return

    gold_df = drop_features_column(gold_df).dropna(subset=["sentence"])
    gold_df = combine_wh_qu(gold_df)
    gold_df = gold_df.drop_duplicates(subset="sentence")

    # ---- 2. Discover model & rationale sheets ----
    model_sheets: dict[str, pd.DataFrame] = {}
    rationale_sheets: dict[str, pd.DataFrame] = {}

    for sheet_name, df_raw in sheets.items():
        # Skip Gold and Excel temporary/hidden sheets
        if sheet_name == "Gold":
            continue
        if sheet_name.startswith("~$"):
            continue

        # Rationales sheets: <model>_rationales
        if sheet_name.endswith("_rationales"):
            base_name = sheet_name[:-11]  # strip "_rationales"
            rationale_sheets[base_name] = df_raw
            continue

        # Only treat BERT_*/GPT_* as model prediction sheets
        name_up = sheet_name.upper()
        if not (name_up.startswith("GPT_") or name_up.startswith("BERT_") or name_up.startswith("ROBERTA_") or name_up.startswith("MODERNBERT_")):
            continue


        df = drop_features_column(df_raw).dropna(subset=["sentence"])
        df = combine_wh_qu(df)
        df = df.drop_duplicates(subset="sentence")

        model_sheets[sheet_name] = df

    if not model_sheets:
        print(f"[WARN] No model sheets (GPT_*/BERT_*) found in {file_path}.")
        return

    # ---- 3. Evaluate each model sheet against Gold ----
    eval_results: list[pd.DataFrame] = []
    per_model_errors: dict[str, pd.DataFrame] = {}

    for sheet_name, model_df in model_sheets.items():
        feature_set = detect_feature_set(sheet_name, model_df)
        if feature_set is None:
            print(f"[SKIP] {sheet_name}: could not determine feature set.")
            continue

        features = MASIS_FEATURES if feature_set == "masis" else EXTENDED_FEATURES

        # binarize probabilities (BERT / RoBERTa / etc) safely
        model_df = binarize_if_probabilistic(model_df, features, sheet_name)
        if any(k in sheet_name.upper() for k in ["BERT", "ROBERTA", "MODERNBERT"]):
            for feat in features:
                if feat in model_df.columns:
                    u = set(model_df[feat].dropna().unique().tolist())
                    if not u.issubset({0,1}):
                        print(f"[WARN] {sheet_name} feature {feat} not binary after binarize: {sorted(list(u))[:10]}")
                        break
        model_sheets[sheet_name] = model_df

        print(f"\n--- Evaluating model sheet: {sheet_name} "
            f"(feature set: {feature_set}, {len(features)} features) ---")

        model_eval = evaluate_model(
            model_df=model_df,
            truth_df=gold_df,
            model_name=sheet_name,
            features=features,
            output_base=output_base,
        )


        if not model_eval.empty:
            eval_results.append(model_eval)

            # Build error dataframe for this model
            err_df = build_error_df(model_df, gold_df, features, model_name=sheet_name, rationale_df=rationale_sheets.get(sheet_name))
            per_model_errors[sheet_name] = err_df

            # Optionally: build annotated rationales if we have them
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


    # If no successful evaluations, nothing else to do
    eval_results = [df for df in eval_results if not df.empty]
    if not eval_results:
        print("[WARN] No non-empty evaluation results.")
        return

    all_eval = pd.concat(eval_results, ignore_index=True)

    for trial in ["A", "B"]:
        sub = all_eval[all_eval["trial"] == trial].copy()
        if sub.empty:
            continue

        eval_dfs_trial = [df for df in eval_results if parse_factors(df["model"].iloc[0])["trial"] == trial]
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



    # factor columns for ALL models (BERT included; may be None)
    factors_df = all_eval["model"].apply(lambda x: pd.Series(parse_factors(x)))
    all_eval = pd.concat([all_eval, factors_df], axis=1)

    gpt_eval = all_eval[all_eval["model"].str.upper().str.startswith("GPT_")].copy()
    gpt_eval = gpt_eval.dropna(subset=["feature_set", "ctx"])

    # include instr/trial if present; function will skip if not exactly 2 levels

    auto_pairwise_deltas(
        gpt_eval,
        factors=["feature_set", "ctx", "instr", "trial"],
        output_base=output_base,
        prefix="GPT_",
    )

    auto_pairwise_deltas(
        all_eval.dropna(subset=["feature_set", "ctx"]),
        factors=["feature_set", "ctx"],
        output_base=output_base,
        prefix="ALLMODELS_",
    )



    if gpt_eval.empty:
        print("[WARN] No GPT_* model rows found; skipping GPT-only analyses (means/vars/deltas).")
    else:
        # ---- GPT means / vars ----
        mean_f1_per_feature = (
            gpt_eval.groupby("feature")["f1"]
            .mean()
            .sort_values(ascending=False)
            .reset_index(name="mean_f1_gpt")
        )
        mean_f1_per_feature.to_csv(os.path.join(output_base, "gpt_mean_f1_per_feature.csv"), index=False)

        var_f1_per_feature = (
            gpt_eval.groupby("feature")["f1"]
            .var(ddof=1)
            .reset_index(name="var_f1_gpt")
        )
        var_f1_per_feature.to_csv(os.path.join(output_base, "gpt_var_f1_per_feature.csv"), index=False)

        # ---- shared features across all GPT models (evenness) ----
        shared = None
        for _, sub in gpt_eval.groupby("model"):
            feats = set(sub["feature"].unique())
            shared = feats if shared is None else (shared & feats)
        shared = shared or set()

        gpt_shared = gpt_eval[gpt_eval["feature"].isin(shared)].copy()

        # ---- ZS -> FSCOT deltas ----
        delta_instr = compute_pairwise_deltas(
            gpt_shared,
            index_cols=["feature", "feature_set"],
            factor_col="instr",
            a_level="ZS",
            b_level="FSCOT",
            metrics=("precision", "recall", "f1"),
            require_metric="f1",
            aggfunc="mean",   # <- future-proof for repeats/seeds
        )
        delta_instr.to_csv(os.path.join(output_base, "delta_ZS_to_FSCOT_per_feature.csv"), index=False)

        # ---- noCTX -> CTX deltas ----
        delta_ctx = compute_pairwise_deltas(
            gpt_shared,
            index_cols=["feature", "feature_set", "instr"],
            factor_col="ctx",
            a_level="noCTX",
            b_level="CTX",
            metrics=("precision", "recall", "f1"),
            require_metric="f1",
            aggfunc="mean",   # <- future-proof for repeats/seeds
        )

        out_csv = os.path.join(output_base, "delta_CTX_per_feature.csv")
        delta_ctx.to_csv(out_csv, index=False)
        print(f"[INFO] Wrote {out_csv} ({len(delta_ctx)} rows)")


    # ---- 4. Cross-model COMPARISONS (even, intersection-based) ----
    # Use intersection alignment so we only compare on features
    # that are shared across all evaluated models.
    all_models_label = "ALL_MODELS"

    # (a) F1 by feature and model (bar + heatmap, intersection)
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

    # (b) Overall F1 (macro over shared features only)
    plot_overall_f1_scores(
        eval_dfs=eval_results,
        align="intersection",
        save_path=os.path.join(output_base, f"{all_models_label}_Overall_F1_intersection.png"),
    )

    # (b2) Overall F1 — FULL per model (each model’s evaluated features)
    plot_overall_f1_scores(
        eval_dfs=eval_results,
        align="union",
        save_path=os.path.join(output_base, f"{all_models_label}_Overall_F1_full_union.png"),
    )

    # (c) Per-feature F1 scores (heatmap) over shared features only
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


    # ---- 5. Error aggregation across models ----
    all_errors = pd.concat(per_model_errors.values(), ignore_index=True) if per_model_errors else pd.DataFrame()

    if not all_errors.empty:
        # Pivot: errors per feature per model
        err_counts = (
            all_errors
            .groupby(["model", "feature"])
            .size()
            .reset_index(name="errors")
            .pivot(index="feature", columns="model", values="errors")
            .fillna(0)
            .sort_index()
        )

        print("\n=== Error counts (per feature x model) ===")
        print(err_counts)

        # Optional: error heatmap for "even" comparison of where models struggle
        plot_model_metrics(
            err_counts=err_counts.reset_index().melt(id_vars="feature", var_name="model", value_name="errors")
                        if isinstance(err_counts, pd.DataFrame) else None,
            metric="errors",
            style="heatmap",
            title="Errors per Feature by Model",
            annotate_heatmap=True,
            figsize=(12, 8),
            save_path=os.path.join(output_base, f"{all_models_label}_Error_Heatmap.png"),
        )

        # Save detailed errors to Excel
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
    # Glob all Excel files in data/ (skip temp files)
    file_paths = [
        fp for fp in glob.glob("data/*.xlsx")
        if not os.path.basename(fp).startswith("~$")
    ]
    if not file_paths:
        print("[INFO] No .xlsx files found under data/.")
        return

    for file_path in file_paths:
        evaluate_sheets(file_path)



if __name__ == "__main__":
    main()