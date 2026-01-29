#!/usr/bin/env python3
import pandas as pd
import glob

incomplete = []
for csv in glob.glob("*predictions.csv"):
    df = pd.read_csv(csv, nrows=1000)  # Fast check
    rows = sum(1 for _ in open(csv)) 
    idx_col = 'idx' if 'idx' in df.columns else None
    valid_idx = len(df.dropna(subset=[idx_col])) if idx_col else 0
    incomplete.append((csv, rows, valid_idx, valid_idx == rows))
print("BROKEN CSVs (<1009 rows OR bad idx):")
for csv, rows, valid, ok in incomplete:
    if rows < 1009 or not ok:
        print(f"  {csv}: {rows} rows, {valid} valid idx → {'FIX NEEDED' if not ok else 'OK'}")
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