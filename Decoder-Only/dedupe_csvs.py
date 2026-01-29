#!/usr/bin/env python3
import pandas as pd
import glob
import os

for csv_path in glob.glob("*.csv"):
    if "_meta.csv" in csv_path: continue  # Skip meta
    
    print(f"Fixing {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Fix idx column
    if 'idx' not in df.columns or df['idx'].isna().all():
        df.insert(0, 'idx', range(len(df)))
    
    # Dedupe by idx, keep LAST prediction (most recent run wins)
    df_clean = df.drop_duplicates('idx', keep='last')
    
    # Save
    df_clean.to_csv(csv_path, index=False)
    print(f"  {len(df)} → {len(df_clean)} rows (deduped)\n")

print("✅ ALL CSVs fixed!")
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