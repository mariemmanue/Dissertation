import os
import glob
import pandas as pd
import argparse

"""
nlprun -q jag -p standard -r 40G -c 2 \
  -n combine_phi4 \
  -o slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation/Decoder-Only && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python combine_predictions.py Phi-4/data/FullTest_Final PHI4_Combined.csv --prefix PHI4_"

"""

def combine_predictions(input_dir, output_file, prefix_to_strip=None):
    # Ensure input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    # Find all files ending in _predictions.csv inside the input dir
    search_path = os.path.join(input_dir, '*_predictions.csv')
    files = glob.glob(search_path)
    
    if not files:
        print(f"No files found matching: {search_path}")
        return

    print(f"Found {len(files)} prediction files in '{input_dir}'")

    all_dfs = []
    
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for file_path in sorted(files):
        try:
            filename = os.path.basename(file_path)
            
            # Skip the output file itself if it's in the same folder
            if filename == os.path.basename(output_file):
                continue

            # Read the CSV
            df = pd.read_csv(file_path)
            
            # Determine the condition name
            # 1. Remove '_predictions.csv' suffix
            condition_name = filename.replace('_predictions.csv', '')
            
            # 2. Remove model prefix if provided (e.g., 'PHI4_')
            if prefix_to_strip and condition_name.startswith(prefix_to_strip):
                condition_name = condition_name[len(prefix_to_strip):]
            
            # Add metadata columns
            df.insert(0, 'experiment_condition', condition_name)
            df['source_filename'] = filename
            
            all_dfs.append(df)
            print(f"  -> Loaded {len(df)} rows from {filename}")
            
        except Exception as e:
            print(f"  !! Error reading {filename}: {e}")

    # Combine everything
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"\nSuccess! Saved {len(combined_df)} total rows to:\n{output_file}")
    else:
        print("No data was combined.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine prediction CSVs into one master sheet.")
    
    # Positional arguments
    parser.add_argument("input_dir", help="Directory containing the CSV files")
    parser.add_argument("output_file", help="Full path for the output CSV file")
    
    # Optional argument
    parser.add_argument("--prefix", help="Text to strip from the start of file names (e.g. 'PHI4_')", default=None)

    args = parser.parse_args()
    
    combine_predictions(args.input_dir, args.output_file, args.prefix)
