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


def combine_wh_qu(df):
    """
    Combines wh_qu1 and wh_qu2 into a single wh_qu feature.
    """
    if 'wh_qu1' in df.columns and 'wh_qu2' in df.columns:
        df["wh_qu"] = df[["wh_qu1", "wh_qu2"]].max(axis=1)
        df = df.drop(columns=["wh_qu1", "wh_qu2"])
    return df



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
    skipped_features = 0

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

        cm_data.append([TP, FP, FN, TN])

    # Convert to DataFrame and plot confusion matrix for per-feature counts
    cm_df = pd.DataFrame(cm_data, columns=["TP", "FP", "FN", "TN"], index=available_features)
    plot_per_feature_confusion_matrix(cm_df, model_name, available_features, save_path=os.path.join(output_base, f'{model_name}_confusion_matrix.png'))

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


def plot_aggregated_confusion_matrix(cm_data, model_name, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', xticklabels=["Predicted Positive", "Predicted Negative"], yticklabels=["Actual Positive", "Actual Negative"])
    plt.title(f'Aggregated Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def build_error_df(model_df: pd.DataFrame, gold_df: pd.DataFrame, features: list[str], model_name: str) -> pd.DataFrame:
    model_sub = model_df.copy()
    gold_sub = gold_df.copy()
    
    model_sub = combine_wh_qu(model_sub)
    gold_sub = combine_wh_qu(gold_sub)

    needed_cols = ["sentence"] + [feat for feat in features if feat in model_sub.columns and feat in gold_sub.columns]
    model_sub = model_sub[needed_cols].copy()
    gold_sub = gold_sub[needed_cols].copy()

    merged = gold_sub.merge(model_sub, on="sentence", suffixes=("_gold", "_pred"))

    rows = []
    for _, row in merged.iterrows():
        for feat in features:
            gold_col = f"{feat}_gold"
            pred_col = f"{feat}_pred"

            if gold_col not in row or pred_col not in row:
                continue

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
    gold_df = combine_wh_qu(gold_df)  # Combine wh_qu features in gold

    bert_df_raw = try_load_sheet(sheets, 'BERT')
    if bert_df_raw is not None:
        bert_df = drop_features_column(bert_df_raw).dropna(subset=["sentence"])
        bert_df = combine_wh_qu(bert_df)  # Combine wh_qu features in BERT
    else:
        bert_df = None

    gpt_df1 = try_load_sheet(sheets, 'GPT-Exp1')
    gpt_df2 = try_load_sheet(sheets, 'GPT-Exp2')
    gpt_df3 = try_load_sheet(sheets, 'GPT-Exp3')  # May be missing

    df_rationales1 = try_load_sheet(sheets, 'rationales-Exp1')
    df_rationales2 = try_load_sheet(sheets, 'rationales-Exp2')
    df_rationales3 = try_load_sheet(sheets, 'rationales-Exp3')  # May be missing

    if bert_df is not None:
        for feat in MASIS_FEATURES:
            if feat in bert_df.columns:
                thr = feat_thresholds.get(feat, 0.5)
                bert_df[feat] = (bert_df[feat].astype(float) >= thr).astype(int)

    # Combine wh_qu1 and wh_qu2 into wh_qu for GPT-Exp1, GPT-Exp2, and GPT-Exp3
    if gpt_df1 is not None:
        gpt_df1 = combine_wh_qu(gpt_df1)
    if gpt_df2 is not None:
        gpt_df2 = combine_wh_qu(gpt_df2)
    if gpt_df3 is not None:
        gpt_df3 = combine_wh_qu(gpt_df3)

    # Evaluate models if data is available
    bert_eval = evaluate_model(bert_df, gold_df, "BERT", MASIS_FEATURES, output_base) if bert_df is not None else pd.DataFrame()
    gpt_eval1 = evaluate_model(gpt_df1, gold_df, "GPT-17", MASIS_FEATURES, output_base) if gpt_df1 is not None else pd.DataFrame()
    gpt_eval2 = evaluate_model(gpt_df2, gold_df, "GPT-24", EXTENDED_FEATURES, output_base) if gpt_df2 is not None else pd.DataFrame()
    gpt_eval3 = evaluate_model(gpt_df3, gold_df, "GPT-24+context", EXTENDED_FEATURES, output_base) if gpt_df3 is not None else pd.DataFrame()

    # Build and save annotated rationales for each experiment
    if gpt_df1 is not None and df_rationales1 is not None:
        annotated_rationales1 = build_annotated_rationales(gpt_df1, df_rationales1, gold_df, MASIS_FEATURES)
        annotated_rationales1.to_csv(os.path.join(output_base, 'GPT-Exp1_rationales.csv'), index=False)

    if gpt_df2 is not None and df_rationales2 is not None:
        annotated_rationales2 = build_annotated_rationales(gpt_df2, df_rationales2, gold_df, EXTENDED_FEATURES)
        annotated_rationales2.to_csv(os.path.join(output_base, 'GPT-Exp2_rationales.csv'), index=False)

    if gpt_df3 is not None and df_rationales3 is not None:
        annotated_rationales3 = build_annotated_rationales(gpt_df3, df_rationales3, gold_df, EXTENDED_FEATURES)
        annotated_rationales3.to_csv(os.path.join(output_base, 'GPT-Exp3_rationales.csv'), index=False)

    # Save predictions for each experiment
    if gpt_df1 is not None:
        gpt_df1.to_csv(os.path.join(output_base, 'GPT-Exp1_predictions.csv'), index=False)
    if gpt_df2 is not None:
        gpt_df2.to_csv(os.path.join(output_base, 'GPT-Exp2_predictions.csv'), index=False)
    if gpt_df3 is not None:
        gpt_df3.to_csv(os.path.join(output_base, 'GPT-Exp3_predictions.csv'), index=False)

    print(f"Completed evaluation for file: {file_path}")

    # Generate combined comparison plots across all models
    all_evals = [bert_eval, gpt_eval1, gpt_eval2, gpt_eval3]
    
    plot_model_metrics(eval_dfs=all_evals, metric="f1", style="bar", save_path=os.path.join(output_base, "All_Models_f1_bar.png"))
    plot_model_metrics(eval_dfs=all_evals, metric="f1", style="heatmap", save_path=os.path.join(output_base, "All_Models_f1_heatmap.png"))

    # Generate error comparison
    bert_exp1_errors = build_error_df(bert_df, gold_df, MASIS_FEATURES, "BERT") if bert_df is not None else pd.DataFrame()
    gpt_exp1_errors = build_error_df(gpt_df1, gold_df, MASIS_FEATURES, "GPT-17") if gpt_df1 is not None else pd.DataFrame()
    gpt_exp2_errors = build_error_df(gpt_df2, gold_df, EXTENDED_FEATURES, "GPT-24") if gpt_df2 is not None else pd.DataFrame()
    gpt_exp3_errors = build_error_df(gpt_df3, gold_df, EXTENDED_FEATURES, "GPT-24+context") if gpt_df3 is not None else pd.DataFrame()

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