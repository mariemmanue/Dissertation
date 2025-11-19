# evaluate.py

"""nlprun -q jag -p standard -r 8G -c 2 -t 0-2 \
  -n eval-aae-results \
  -o slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation/Decoder-Only/GPT && \
   mkdir -p slurm_logs data/results && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   export HF_HOME=/nlp/scr/mtano/hf_home && \
   python evaluate.py" """

"""nlprun -q jag -p standard -r 8G -c 2 -t 0-5 \
  -n gpt-eval-aae \
  -o slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation/Decoder-Only/GPT && \
   mkdir -p slurm_logs data/results && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   export HF_HOME=/nlp/scr/mtano/hf_home && \
   python gpt_experiments.py --file data/Run2.xlsx --sheet GPT-Exp1 --extended --context && \
   python evaluate.py" """

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import List, Optional, Literal
from gpt_experiments import (
    EXTENDED_FEATURES,
    MASIS_FEATURES,
)

feat_thresholds = {
    "multiple-neg": 0.5,  # Example threshold
}

output_dir = "data/results"
os.makedirs(output_dir, exist_ok=True)


def try_load_sheet(sheets, sheet_name):
    return sheets[sheet_name] if sheet_name in sheets else None


def drop_features_column(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=['FEATURES', 'Source'], errors='ignore')


def evaluate_model(model_df, truth_df, model_name, features):
    print(f"\n=== {model_name} Evaluation ===")

    if model_df is None:
        print(f"{model_name} data not available. Skipping evaluation.")
        return pd.DataFrame()

    model_df = model_df.drop_duplicates(subset='sentence')
    truth_df = truth_df.drop_duplicates(subset='sentence')

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
    skipped_features = 0

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
            continue

        y_true = merged['y_true'].astype(int)
        y_pred = merged['y_pred'].astype(int)

        if y_true.sum() == 0 and y_pred.sum() == 0:
            print(f"Skipping {feat} — no positive instances in ground truth and predictions.")
            skipped_features += 1
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

            gold_v = int(row[gold_col])
            pred_v = int(row[pred_col])

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

def build_error_df(model_df: pd.DataFrame,
                   gold_df: pd.DataFrame,
                   features: list[str],
                   model_name: str) -> pd.DataFrame:
    """
    model_df: predictions with columns ['sentence'] + features (0/1)
    gold_df:  gold labels with same columns
    features: list of feature names to compare
    model_name: label to store in the 'model' column
    """
    # make sure we only keep the columns we care about
    needed_cols = ["sentence"] + features
    model_sub = model_df[needed_cols].copy()
    gold_sub  = gold_df[needed_cols].copy()

    merged = gold_sub.merge(model_sub, on="sentence", suffixes=("_gold", "_pred"))

    rows = []
    for _, row in merged.iterrows():
        for feat in features:
            gold_col = f"{feat}_gold"
            pred_col = f"{feat}_pred"

            gold_val = row.get(gold_col, None)
            pred_val = row.get(pred_col, None)

            if pd.isna(gold_val) or pd.isna(pred_val):
                continue

            if int(gold_val) != int(pred_val):
                rows.append({
                    "model": model_name,
                    "sentence": row["sentence"],
                    "feature": feat,
                    "gold": int(gold_val),
                    "pred": int(pred_val),
                })

    return pd.DataFrame(rows)


def evaluate_sheets(file_path):
    sheets = pd.read_excel(file_path, sheet_name=None)
    
    output_base = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0])
    os.makedirs(output_base, exist_ok=True)

    gold_df = try_load_sheet(sheets, 'Gold')
    if gold_df is None:  # Ensure 'Gold' sheet exists
        print(f"Gold sheet not found in {file_path}")
        return
    gold_df = drop_features_column(gold_df).dropna(subset=["sentence"])

    bert_df_raw = try_load_sheet(sheets, 'BERT')
    bert_df = drop_features_column(bert_df_raw).dropna(subset=["sentence"]) if bert_df_raw is not None else None

    gpt_df1 = try_load_sheet(sheets, 'GPT-Exp1')
    gpt_df2 = try_load_sheet(sheets, 'GPT-Exp2')
    gpt_df3 = try_load_sheet(sheets, 'GPT-Exp3')
    
    df_rationales1 = try_load_sheet(sheets, 'rationales-Exp1')
    df_rationales2 = try_load_sheet(sheets, 'rationales-Exp2')
    df_rationales3 = try_load_sheet(sheets, 'rationales-Exp3')

    if bert_df is not None:
        for feat in MASIS_FEATURES:
            if feat in bert_df.columns:
                thr = feat_thresholds.get(feat, 0.5)
                bert_df[feat] = (bert_df[feat].astype(float) >= thr).astype(int)

    # Evaluate models if data is available
    bert_eval = evaluate_model(bert_df, gold_df, "BERT", MASIS_FEATURES) if bert_df is not None else pd.DataFrame()
    gpt_eval1 = evaluate_model(gpt_df1, gold_df, "GPT-17", MASIS_FEATURES) if gpt_df1 is not None else pd.DataFrame()
    gpt_eval2 = evaluate_model(gpt_df2, gold_df, "GPT-24", EXTENDED_FEATURES) if gpt_df2 is not None else pd.DataFrame()
    gpt_eval3 = evaluate_model(gpt_df3, gold_df, "GPT-24+context", EXTENDED_FEATURES) if gpt_df3 is not None else pd.DataFrame()

    # Build and save annotated rationales for each experiment
    annotated_rationales1 = build_annotated_rationales(gpt_df1, df_rationales1, gold_df, MASIS_FEATURES)
    annotated_rationales1.to_csv(os.path.join(output_base, 'GPT-Exp1_rationales.csv'), index=False)

    annotated_rationales2 = build_annotated_rationales(gpt_df2, df_rationales2, gold_df, EXTENDED_FEATURES)
    annotated_rationales2.to_csv(os.path.join(output_base, 'GPT-Exp2_rationales.csv'), index=False)

    annotated_rationales3 = build_annotated_rationales(gpt_df3, df_rationales3, gold_df, EXTENDED_FEATURES)
    annotated_rationales3.to_csv(os.path.join(output_base, 'GPT-Exp3_rationales.csv'), index=False)

    # Save predictions for each experiment
    gpt_df1.to_csv(os.path.join(output_base, 'GPT-Exp1_predictions.csv'), index=False)
    gpt_df2.to_csv(os.path.join(output_base, 'GPT-Exp2_predictions.csv'), index=False)
    gpt_df3.to_csv(os.path.join(output_base, 'GPT-Exp3_predictions.csv'), index=False)

    # Plot metrics: Comparing BERT and GPT Experiments
    plot_model_metrics(eval_dfs=[gpt_eval1, bert_eval], metric="f1", style="bar", save_path=os.path.join(output_base, "GPT1_vs_BERT_f1_bar.png"))
    plot_model_metrics(eval_dfs=[gpt_eval1, bert_eval], metric="f1", style="heatmap", save_path=os.path.join(output_base, "GPT1_vs_BERT_f1_heatmap.png"))

    plot_model_metrics(eval_dfs=[gpt_eval1, gpt_eval2], metric="f1", style="bar", align="intersection", save_path=os.path.join(output_base, "GPT1_vs_GPT2_f1_bar.png"))
    plot_model_metrics(eval_dfs=[gpt_eval1, gpt_eval2], metric="f1", style="heatmap", align="intersection", save_path=os.path.join(output_base, "GPT1_vs_GPT2_f1_heatmap.png"))

    plot_model_metrics(eval_dfs=[gpt_eval2, gpt_eval3], metric="f1", style="bar", save_path=os.path.join(output_base, "GPT2_vs_GPT3_f1_bar.png"))
    plot_model_metrics(eval_dfs=[gpt_eval2, gpt_eval3], metric="f1", style="heatmap", save_path=os.path.join(output_base, "GPT2_vs_GPT3_f1_heatmap.png"))

    print(f"Completed evaluation for file: {file_path}")

    # Generate error comparison
    bert_exp1_errors = build_error_df(bert_df, gold_df, MASIS_FEATURES, "BERT")
    gpt_exp1_errors = build_error_df(gpt_df1, gold_df, MASIS_FEATURES, "GPT-17")
    gpt_exp2_errors = build_error_df(gpt_df2, gold_df, EXTENDED_FEATURES, "GPT-24")
    gpt_exp3_errors = build_error_df(gpt_df3, gold_df, EXTENDED_FEATURES, "GPT-24+context")

    all_errors = pd.concat([gpt_exp1_errors, bert_exp1_errors, gpt_exp2_errors, gpt_exp3_errors], ignore_index=True)

    # Quick pivot: errors per feature per model
    err_counts = (
        all_errors
        .groupby(["model", "feature"])
        .size()
        .reset_index(name="error_count")
        .pivot(index="feature", columns="model", values="error_count")
        .fillna(0)
        .sort_index()
    )

    print(err_counts)

    with pd.ExcelWriter(os.path.join(output_base, "model_errors_all_experiments.xlsx")) as writer:
        gpt_exp1_errors.to_excel(writer, sheet_name="GPT-Exp1_errors", index=False)
        bert_exp1_errors.to_excel(writer, sheet_name="BERT_errors", index=False)
        gpt_exp2_errors.to_excel(writer, sheet_name="GPT-Exp2_errors", index=False)
        gpt_exp3_errors.to_excel(writer, sheet_name="GPT-Exp3_errors", index=False)
        all_errors.to_excel(writer, sheet_name="all_errors", index=False)
        err_counts.to_excel(writer, sheet_name="error_counts_pivot")

def main():
    file_paths = glob.glob("data/*.xlsx")
    for file_path in file_paths:
        evaluate_sheets(file_path)

if __name__ == "__main__":
    main()

