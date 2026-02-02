import pandas as pd
import glob
import os

# Directory to scan
# Adjusted to look in the specific folder you mentioned
TARGET_DIR = "Gemini/data/results/FullTest_Final"
search_pattern = os.path.join(TARGET_DIR, "*_predictions.csv")

print(f"Looking for files in: {search_pattern}")
files_to_fix = glob.glob(search_pattern)

if not files_to_fix:
    print("No files found! Check the path.")
else:
    print(f"Found {len(files_to_fix)} files. Starting process...\n")

for csv_path in files_to_fix:
    print(f"Processing: {os.path.basename(csv_path)}...")
    
    # Read without header initially to see raw structure if it's messed up
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
    except Exception as e:
        print(f"  [ERROR] Read failed: {e}")
        continue

    if df.empty:
        print("  [SKIP] File is empty.")
        continue

    initial_count = len(df)
    print(f"  Rows before: {initial_count}")

    # Process only if 'idx' exists
    if 'idx' in df.columns:
        # Ensure idx is numeric
        df['idx'] = pd.to_numeric(df['idx'], errors='coerce')
        
        # Drop rows where idx is NaN (bad lines that slipped through)
        df = df.dropna(subset=['idx'])
        
        # Cast to integer
        df['idx'] = df['idx'].astype(int)
        
        # Drop duplicates by idx, keeping the LAST valid run (simulating a resume)
        df = df.drop_duplicates(subset=['idx'], keep='last')
        
        # Sort strictly by idx
        df = df.sort_values('idx')
        
        final_count = len(df)
        print(f"  Rows after dedupe: {final_count} (Removed {initial_count - final_count})")
        
        if not df.empty:
            print(f"  Range: {df['idx'].min()} to {df['idx'].max()}")
        
        # Save back to the same path
        df.to_csv(csv_path, index=False)
        print("  [SUCCESS] Saved clean CSV.\n")
    else:
        print("  [ERROR] No 'idx' column found! Skipping.\n")

print("All files processed.")
