import os
import glob
import pandas as pd
import argparse

"""
# ── Phi-4 ──
nlprun -q jag -p standard -r 40G -c 2 -n combine_phi4 \
  -o Decoder-Only/Phi-4/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && conda activate cgedit && \
   python Decoder-Only/combine_predictions.py Decoder-Only/Phi-4/data/FullTest_Final/ Decoder-Only/Phi-4_Combined.xlsx"

# ── Phi-4-reasoning ──
nlprun -q jag -p standard -r 40G -c 2 -n combine_phi4r \
  -o Decoder-Only/Phi-4-reasoning/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && conda activate cgedit && \
   python Decoder-Only/combine_predictions.py Decoder-Only/Phi-4-reasoning/data/FullTest_Final/ Decoder-Only/Phi-4-reasoning_Combined.xlsx"

# ── Gemini 2.5 Flash ──
nlprun -q jag -p standard -r 40G -c 2 -n combine_gemini \
  -o Decoder-Only/Gemini/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && conda activate cgedit && \
   python Decoder-Only/combine_predictions.py Decoder-Only/Gemini/data/FullTest_Final/ Decoder-Only/Gemini_Combined.xlsx"

# ── Gemini 3 Pro ──
nlprun -q jag -p standard -r 40G -c 2 -n combine_gem3pro \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && conda activate cgedit && \
   python Decoder-Only/combine_predictions.py Decoder-Only/Gemini3_Pro/data/FullTest_Final/ Decoder-Only/Gemini3_Pro_Combined.xlsx"

# ── GPT-4.1 ──
nlprun -q jag -p standard -r 40G -c 2 -n combine_gpt41 \
  -o Decoder-Only/GPT41/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && conda activate cgedit && \
   python Decoder-Only/combine_predictions.py Decoder-Only/GPT41/data/FullTest_Final/ Decoder-Only/GPT41_Combined.xlsx"

# ── GPT-5.2 Instant ──
nlprun -q jag -p standard -r 40G -c 2 -n combine_gpt52i \
  -o Decoder-Only/GPT52_Instant/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && conda activate cgedit && \
   python Decoder-Only/combine_predictions.py Decoder-Only/GPT52_Instant/data/FullTest_Final/ Decoder-Only/GPT52_Instant_Combined.xlsx"

# ── GPT-5.2 Thinking (medium) ──
nlprun -q jag -p standard -r 40G -c 2 -n combine_gpt52tm \
  -o Decoder-Only/GPT52_Thinking_Med/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && conda activate cgedit && \
   python Decoder-Only/combine_predictions.py Decoder-Only/GPT52_Thinking_Med/data/FullTest_Final/ Decoder-Only/GPT52_Thinking_Med_Combined.xlsx"

# ── GPT-5.2 Thinking (high) ──
nlprun -q jag -p standard -r 40G -c 2 -n combine_gpt52th \
  -o Decoder-Only/GPT52_Thinking_High/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && conda activate cgedit && \
   python Decoder-Only/combine_predictions.py Decoder-Only/GPT52_Thinking_High/data/FullTest_Final/ Decoder-Only/GPT52_Thinking_High_Combined.xlsx"

# ── Qwen 2.5-7B ──
nlprun -q jag -p standard -r 40G -c 2 -n combine_qwen25 \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && conda activate cgedit && \
   python Decoder-Only/combine_predictions.py Decoder-Only/Qwen2.5/data/FullTest_Final/ Decoder-Only/Qwen2.5_Combined.xlsx"

# ── Qwen3-32B (non-thinking) ──
nlprun -q jag -p standard -r 40G -c 2 -n combine_qwen3 \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && conda activate cgedit && \
   python Decoder-Only/combine_predictions.py Decoder-Only/Qwen3-32B/data/FullTest_Final/ Decoder-Only/Qwen3-32B_Combined.xlsx"

# ── Qwen3-32B (thinking) ──
nlprun -q jag -p standard -r 40G -c 2 -n combine_qwen3t \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && conda activate cgedit && \
   python Decoder-Only/combine_predictions.py Decoder-Only/Qwen3-32B-Thinking/data/FullTest_Final/ Decoder-Only/Qwen3-32B-Thinking_Combined.xlsx"

# ── QwQ-32B ──
nlprun -q jag -p standard -r 40G -c 2 -n combine_qwq \
  -o Decoder-Only/QwQ-32B/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && conda activate cgedit && \
   python Decoder-Only/combine_predictions.py Decoder-Only/QwQ-32B/data/FullTest_Final/ Decoder-Only/QwQ-32B_Combined.xlsx"

# ── Llama 3.1-70B ──
nlprun -q jag -p standard -r 40G -c 2 -n combine_llama70b \
  -o Decoder-Only/Llama-3.1-70B/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && conda activate cgedit && \
   python Decoder-Only/combine_predictions.py Decoder-Only/Llama-3.1-70B/data/FullTest_Final/ Decoder-Only/Llama-3.1-70B_Combined.xlsx"
"""

def _detect_common_prefix(names):
    """Auto-detect a shared model prefix like 'GPT41_' or 'GEMINI_' from a list of config names."""
    if not names:
        return None
    # Split each name by '_' and find the longest common leading tokens
    splits = [n.split("_") for n in names]
    prefix_tokens = []
    for parts in zip(*splits):
        if len(set(p.upper() for p in parts)) == 1:
            prefix_tokens.append(parts[0])
        else:
            break
    if prefix_tokens:
        return "_".join(prefix_tokens) + "_"
    return None


def combine_to_excel(input_dir, output_file, prefix_to_strip=None):
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    if not output_file.endswith('.xlsx'):
        output_file = os.path.splitext(output_file)[0] + '.xlsx'
        print(f"Note: Output file changed to '{output_file}' (Sheets require Excel format)")

    search_path = os.path.join(input_dir, '*_predictions.csv')
    files = sorted(glob.glob(search_path))

    if not files:
        print(f"No files found matching: {search_path}")
        return

    print(f"Found {len(files)} prediction files. Creating Excel workbook...")

    # Auto-detect prefix if not provided
    full_names = [os.path.basename(f).replace('_predictions.csv', '') for f in files]
    if prefix_to_strip is None:
        prefix_to_strip = _detect_common_prefix(full_names)
    if prefix_to_strip:
        print(f"Stripping prefix: '{prefix_to_strip}'")

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            files_written = 0

            for file_path in files:
                try:
                    filename = os.path.basename(file_path)
                    df = pd.read_csv(file_path)

                    full_name = filename.replace('_predictions.csv', '')
                    # model_name column keeps the FULL name for traceability
                    df.insert(0, 'model_name', full_name)

                    # Sheet tab uses the stripped name (config only, no model prefix)
                    if prefix_to_strip and full_name.upper().startswith(prefix_to_strip.upper()):
                        sheet_name = full_name[len(prefix_to_strip):]
                    else:
                        sheet_name = full_name

                    if len(sheet_name) > 31:
                        original_name = sheet_name
                        sheet_name = sheet_name[:31]
                        print(f"  Warning: Truncated '{original_name}' to '{sheet_name}' (Excel 31-char limit)")

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
