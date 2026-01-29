#!/usr/bin/env python3
import os, glob, re
import pandas as pd

EXCEL_PATH = "FullTest_Final.xlsx"
RESULTS_DIR = "Gemini/data/results/FullTest_Final"

def norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    # normalize common “smart quotes”
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    return s

# Load canonical order from Gold sheet
sheets = pd.read_excel(EXCEL_PATH, sheet_name=None)
if "Gold" not in sheets:
    raise ValueError("Excel must contain a sheet named 'Gold' (as expected by the pipeline).")
gold_df = sheets["Gold"]
sent_col = "sentence" if "sentence" in gold_df.columns else gold_df.columns[0]
gold_sents = [norm(x) for x in gold_df[sent_col].dropna().tolist()]

# Build mapping: sentence -> list of positions (handles duplicates safely)
pos_map = {}
for i, s in enumerate(gold_sents):
    pos_map.setdefault(s, []).append(i)

def assign_idx_by_sentence(df: pd.DataFrame, sentence_col: str) -> pd.DataFrame:
    used = {s: 0 for s in pos_map.keys()}
    new_idx = []
    unmatched = 0

    for s in df[sentence_col].tolist():
        ns = norm(s)
        if ns in pos_map:
            k = used[ns]
            if k < len(pos_map[ns]):
                new_idx.append(pos_map[ns][k])
                used[ns] += 1
            else:
                # more occurrences in CSV than in Gold
                new_idx.append(-1)
                unmatched += 1
        else:
            new_idx.append(-1)
            unmatched += 1

    out = df.copy()
    out["idx"] = new_idx
    return out, unmatched

csvs = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.csv")))
print(f"Found {len(csvs)} CSV files in {RESULTS_DIR}")

for path in csvs:
    name = os.path.basename(path)
    try:
        df = pd.read_csv(path, on_bad_lines="skip", dtype=str)
        if len(df) == 0:
            print(f"{name}: empty, skip")
            continue

        # Identify sentence column robustly (pipeline writes: idx, sentence, ...) 
        # but some files may have lost/changed headers.
        if "sentence" in df.columns:
            sentence_col = "sentence"
        elif len(df.columns) >= 2 and df.columns[0].lower() == "idx":
            sentence_col = df.columns[1]
        else:
            # fallback: assume first column is sentence if no clear idx
            sentence_col = df.columns[0]

        # If idx exists and is clean numeric, just sort by it.
        idx_ok = False
        if "idx" in df.columns:
            try:
                _ = pd.to_numeric(df["idx"], errors="raise")
                idx_ok = True
            except Exception:
                idx_ok = False

        if not idx_ok:
            df2, unmatched = assign_idx_by_sentence(df, sentence_col)
        else:
            df2 = df.copy()
            df2["idx"] = pd.to_numeric(df2["idx"], errors="coerce").fillna(-1).astype(int)
            unmatched = int((df2["idx"] < 0).sum())

        # Dedupe by idx (keep last = latest run wins), then sort by idx
        df2 = df2.drop_duplicates(subset=["idx"], keep="last")
        df2 = df2.sort_values("idx", kind="stable")

        # Put idx, sentence first (nice + consistent with writer.writerow(idx, sentence, ...))
        if sentence_col != "sentence" and "sentence" not in df2.columns:
            df2 = df2.rename(columns={sentence_col: "sentence"})
        cols = list(df2.columns)
        cols = ["idx"] + (["sentence"] if "sentence" in cols else []) + [c for c in cols if c not in ("idx","sentence")]
        df2 = df2[cols]

        df2.to_csv(path, index=False)
        print(f"{name}: wrote {len(df2)} rows (unmatched={unmatched})")

    except Exception as e:
        print(f"{name}: ERROR {e}")

print("Done.")
# #!/usr/bin/env python3
# import pandas as pd
# import glob
# import os
# import warnings
# warnings.filterwarnings("ignore")

# csv_dir = "Gemini/data/results/FullTest_Final/"
# csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
# print(f"Found {len(csv_files)} CSV files")

# for csv_path in csv_files:
#     print(f"Processing {os.path.basename(csv_path)}...")
    
#     try:
#         # Robust CSV read (handles malformed rows)
#         df = pd.read_csv(csv_path, on_bad_lines='skip')
        
#         if len(df) == 0:
#             print(f"  Empty file, skipping")
#             continue
            
#         # Assume first column is sentence/utterance (handles both idx and sentence naming)
#         sentence_col = df.columns[0]
#         print(f"  Using column '{sentence_col}' for deduplication")
        
#         # Drop duplicates by sentence, keep last row
#         df_sorted = df.sort_values(sentence_col).drop_duplicates(sentence_col, keep='last')
        
#         # Save back
#         df_sorted.to_csv(csv_path, index=False)
        
#         original_count = len(df)
#         unique_count = len(df_sorted)
#         print(f"  {original_count} → {unique_count} rows ({original_count - unique_count} duplicates removed)")
        
#     except Exception as e:
#         print(f"  ERROR: {e} - skipping")

# print("✅ All CSVs deduplicated by sentence (keeping latest predictions)")