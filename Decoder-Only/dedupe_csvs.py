#!/usr/bin/env python3
import pandas as pd
import glob
import os

# Directory with your CSVs
csv_dir = "/nlp/scr/mtano/Dissertation/Decoder-Only/Gemini/data/results/FullTest_Final/"

# Find all CSV files
csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

print(f"Found {len(csv_files)} CSV files")

for csv_path in csv_files:
    print(f"Processing {os.path.basename(csv_path)}...")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Check if 'idx' column exists
    if 'idx' in df.columns:
        # Sort by idx, then drop duplicates keeping last row for each idx
        df_sorted = df.sort_values('idx').drop_duplicates('idx', keep='last')
        
        # Save back (overwrite original)
        df_sorted.to_csv(csv_path, index=False)
        
        # Report
        original_count = len(df)
        unique_count = len(df_sorted)
        print(f"  {original_count} → {unique_count} rows ({original_count - unique_count} duplicates removed)")
    else:
        print(f"  No 'idx' column, skipping")

print("✅ All CSVs deduplicated by idx (keeping latest predictions)")
