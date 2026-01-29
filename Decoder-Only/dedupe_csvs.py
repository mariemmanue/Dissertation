import pandas as pd
import sys

# File to fix
csv_path = "Gemini/data/results/FullTest_Final/GEMINI_25_FSCOT_CTX_two_nolegit_rats_predictions.csv"

print(f"Reading {csv_path}...")
# Read without header initially to see raw structure if it's messed up
try:
    df = pd.read_csv(csv_path, on_bad_lines='skip')
except:
    print("Read failed, trying fallback...")
    sys.exit(1)

print(f"Rows before: {len(df)}")

# Sort by idx first to see what we have
if 'idx' in df.columns:
    # Ensure idx is numeric
    df['idx'] = pd.to_numeric(df['idx'], errors='coerce')
    df = df.dropna(subset=['idx'])
    df['idx'] = df['idx'].astype(int)
    
    # Drop duplicates by idx, keeping the LAST valid run
    df = df.drop_duplicates(subset=['idx'], keep='last')
    
    # Sort
    df = df.sort_values('idx')
    
    print(f"Rows after dedupe: {len(df)}")
    print(f"Min idx: {df['idx'].min()}, Max idx: {df['idx'].max()}")
    
    # Save
    df.to_csv(csv_path, index=False)
    print("Saved clean CSV.")
else:
    print("Error: No 'idx' column found!")