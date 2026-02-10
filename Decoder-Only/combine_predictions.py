import os
import glob
import pandas as pd
import argparse

"""
nlprun -q jag -p standard -r 40G -c 2 \
  -n combine_phi4 \
  -o Decoder-Only/Phi-4/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
    python Decoder-Only/combine_predictions.py \
    Decoder-Only/Phi-4/data/FullTest_Final/ \
    Decoder-Only/Phi-4_Combined.xlsx \
    --prefix PHI4_"

nlprun -q jag -p standard -r 40G -c 2 \
  -n combine_gem \
  -o Decoder-Only/Gemini/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
    python Decoder-Only/combine_predictions.py \
    Decoder-Only/Gemini/data/FullTest_Final/ \
    Decoder-Only/Gemini_Combined.xlsx \
    --prefix GEMINI_"
"""

def combine_to_excel(input_dir, output_file, prefix_to_strip=None):
    # Ensure input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    # Force the output extension to be .xlsx
    if not output_file.endswith('.xlsx'):
        output_file = os.path.splitext(output_file)[0] + '.xlsx'
        print(f"Note: Output file changed to '{output_file}' (Sheets require Excel format)")

    # Find all files ending in _predictions.csv
    search_path = os.path.join(input_dir, '*_predictions.csv')
    files = glob.glob(search_path)
    
    if not files:
        print(f"No files found matching: {search_path}")
        return

    print(f"Found {len(files)} prediction files. Creating Excel workbook...")

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Use ExcelWriter to write multiple sheets to one file
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            files_written = 0
            
            for file_path in sorted(files):
                try:
                    filename = os.path.basename(file_path)
                    
                    # Read the CSV
                    df = pd.read_csv(file_path)
                    
                    # --- NEW CODE: Add Model Name Column ---
                    # Uses the full filename (e.g., 'PHI4_aint_predictions.csv')
                    # You can change 'model_name' to whatever header you prefer
                    df.insert(0, 'model_name', filename.replace('_predictions.csv', ''))
                    # ---------------------------------------

                    # Clean up the name for the Sheet Tab
                    # 1. Remove '_predictions.csv'
                    sheet_name = filename.replace('_predictions.csv', '')
                    
                    # 2. Remove prefix (e.g. 'PHI4_')
                    if prefix_to_strip and sheet_name.startswith(prefix_to_strip):
                        sheet_name = sheet_name[len(prefix_to_strip):]
                    
                    # 3. Excel limits sheet names to 31 characters
                    if len(sheet_name) > 31:
                        original_name = sheet_name
                        sheet_name = sheet_name[:31]
                        print(f"  Warning: Truncated sheet name '{original_name}' to '{sheet_name}' (Excel limit)")
                    
                    # Write to a specific sheet (Tab)
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"  -> Added tab: {sheet_name} ({len(df)} rows)")
                    files_written += 1
                    
                except Exception as e:
                    print(f"  !! Error processing {filename}: {e}")
            
            if files_written > 0:
                print(f"\nSuccess! Saved {files_written} sheets to:\n{output_file}")
            else:
                print("No sheets were written.")
                
    except ImportError:
        print("\nERROR: Missing library 'openpyxl'.")
        print("Please run: pip install openpyxl")
    except Exception as e:
        print(f"\nCritical Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine prediction CSVs into one Excel file with multiple sheets.")
    
    parser.add_argument("input_dir", help="Directory containing the CSV files")
    parser.add_argument("output_file", help="Full path for the output .xlsx file")
    parser.add_argument("--prefix", help="Text to strip from filenames (e.g. 'PHI4_')", default=None)

    args = parser.parse_args()
    
    combine_to_excel(args.input_dir, args.output_file, args.prefix)
