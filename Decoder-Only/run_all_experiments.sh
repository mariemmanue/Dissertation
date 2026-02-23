#!/bin/bash
# Auto-generated experimental run script
# Total jobs: 312
#
# Grid per model:
#   instruction_type:   zero_shot, few_shot, zero_shot_cot, few_shot_cot  (4)
#   context:            off, single_turn, two_turn                        (3)
#   dialect_legitimacy: off, on                                           (2)
#   output_format:      markdown (all models)                             (1)
#   ──────────────────────────────────────────────────────────────────────
#   Total per model:    24
#
# Usage:
#   chmod +x run_all_experiments.sh
#   ./run_all_experiments.sh              # launch all 312 jobs
#   ./run_all_experiments.sh phi4           # launch only phi4 jobs (24)
#   ./run_all_experiments.sh gemini         # launch only gemini jobs (24)
#   ./run_all_experiments.sh gpt52_instant  # launch only GPT-5.2 Instant jobs (24)
#
# Available model keys: phi4, gemini, qwen25_7b, gemini3_pro, gpt41, gpt52_instant, gpt52_think_med, gpt52_think_high, phi4_reasoning, llama70b, qwen3_32b, qwen3_32b_think, qwq_32b

set -e

MODEL_FILTER="${1:-all}"

# ============================================================
# PHI4 — microsoft/phi-4
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "phi4" ]]; then

mkdir -p Decoder-Only/Phi-4/slurm_logs
mkdir -p Decoder-Only/Phi-4/data

echo "[001] Launching: PHI4_ZS_noCTX_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 01_phi4_zs_noctx_noleg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_ZS_noCTX_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt"

echo "[002] Launching: PHI4_ZS_noCTX_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 02_phi4_zs_noctx_leg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_ZS_noCTX_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[003] Launching: PHI4_ZS_CTX1t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 03_phi4_zs_ctx1t_noleg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_ZS_CTX1t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[004] Launching: PHI4_ZS_CTX1t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 04_phi4_zs_ctx1t_leg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_ZS_CTX1t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[005] Launching: PHI4_ZS_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 05_phi4_zs_ctx2t_noleg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_ZS_CTX2t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[006] Launching: PHI4_ZS_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 06_phi4_zs_ctx2t_leg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_ZS_CTX2t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[007] Launching: PHI4_FS_noCTX_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 07_phi4_fs_noctx_noleg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_FS_noCTX_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt"

echo "[008] Launching: PHI4_FS_noCTX_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 08_phi4_fs_noctx_leg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_FS_noCTX_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[009] Launching: PHI4_FS_CTX1t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 09_phi4_fs_ctx1t_noleg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_FS_CTX1t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[010] Launching: PHI4_FS_CTX1t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 10_phi4_fs_ctx1t_leg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_FS_CTX1t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[011] Launching: PHI4_FS_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 11_phi4_fs_ctx2t_noleg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_FS_CTX2t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[012] Launching: PHI4_FS_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 12_phi4_fs_ctx2t_leg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_FS_CTX2t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[013] Launching: PHI4_ZScot_noCTX_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 13_phi4_zscot_noctx_noleg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_ZScot_noCTX_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt"

echo "[014] Launching: PHI4_ZScot_noCTX_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 14_phi4_zscot_noctx_leg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_ZScot_noCTX_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[015] Launching: PHI4_ZScot_CTX1t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 15_phi4_zscot_ctx1t_noleg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_ZScot_CTX1t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[016] Launching: PHI4_ZScot_CTX1t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 16_phi4_zscot_ctx1t_leg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_ZScot_CTX1t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[017] Launching: PHI4_ZScot_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 17_phi4_zscot_ctx2t_noleg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_ZScot_CTX2t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[018] Launching: PHI4_ZScot_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 18_phi4_zscot_ctx2t_leg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_ZScot_CTX2t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[019] Launching: PHI4_FScot_noCTX_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 19_phi4_fscot_noctx_noleg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_FScot_noCTX_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt"

echo "[020] Launching: PHI4_FScot_noCTX_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 20_phi4_fscot_noctx_leg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_FScot_noCTX_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[021] Launching: PHI4_FScot_CTX1t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 21_phi4_fscot_ctx1t_noleg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_FScot_CTX1t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[022] Launching: PHI4_FScot_CTX1t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 22_phi4_fscot_ctx1t_leg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_FScot_CTX1t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[023] Launching: PHI4_FScot_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 23_phi4_fscot_ctx2t_noleg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_FScot_CTX2t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[024] Launching: PHI4_FScot_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 24_phi4_fscot_ctx2t_leg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_FScot_CTX2t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

fi

# ============================================================
# GEMINI — gemini-2.5-flash
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "gemini" ]]; then

mkdir -p Decoder-Only/Gemini/slurm_logs
mkdir -p Decoder-Only/Gemini/data

echo "[025] Launching: GEMINI_ZS_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 01_gemini_zs_noctx_noleg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_ZS_noCTX_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt"

echo "[026] Launching: GEMINI_ZS_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 02_gemini_zs_noctx_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_ZS_noCTX_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[027] Launching: GEMINI_ZS_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 03_gemini_zs_ctx1t_noleg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_ZS_CTX1t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[028] Launching: GEMINI_ZS_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 04_gemini_zs_ctx1t_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_ZS_CTX1t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[029] Launching: GEMINI_ZS_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 05_gemini_zs_ctx2t_noleg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_ZS_CTX2t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[030] Launching: GEMINI_ZS_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 06_gemini_zs_ctx2t_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_ZS_CTX2t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[031] Launching: GEMINI_FS_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 07_gemini_fs_noctx_noleg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_FS_noCTX_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt"

echo "[032] Launching: GEMINI_FS_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 08_gemini_fs_noctx_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_FS_noCTX_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[033] Launching: GEMINI_FS_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 09_gemini_fs_ctx1t_noleg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_FS_CTX1t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[034] Launching: GEMINI_FS_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 10_gemini_fs_ctx1t_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_FS_CTX1t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[035] Launching: GEMINI_FS_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 11_gemini_fs_ctx2t_noleg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_FS_CTX2t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[036] Launching: GEMINI_FS_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 12_gemini_fs_ctx2t_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_FS_CTX2t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[037] Launching: GEMINI_ZScot_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 13_gemini_zscot_noctx_noleg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_ZScot_noCTX_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt"

echo "[038] Launching: GEMINI_ZScot_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 14_gemini_zscot_noctx_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_ZScot_noCTX_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[039] Launching: GEMINI_ZScot_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 15_gemini_zscot_ctx1t_noleg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_ZScot_CTX1t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[040] Launching: GEMINI_ZScot_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 16_gemini_zscot_ctx1t_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_ZScot_CTX1t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[041] Launching: GEMINI_ZScot_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 17_gemini_zscot_ctx2t_noleg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_ZScot_CTX2t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[042] Launching: GEMINI_ZScot_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 18_gemini_zscot_ctx2t_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_ZScot_CTX2t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[043] Launching: GEMINI_FScot_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 19_gemini_fscot_noctx_noleg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_FScot_noCTX_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt"

echo "[044] Launching: GEMINI_FScot_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 20_gemini_fscot_noctx_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_FScot_noCTX_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[045] Launching: GEMINI_FScot_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 21_gemini_fscot_ctx1t_noleg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_FScot_CTX1t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[046] Launching: GEMINI_FScot_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 22_gemini_fscot_ctx1t_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_FScot_CTX1t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[047] Launching: GEMINI_FScot_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 23_gemini_fscot_ctx2t_noleg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_FScot_CTX2t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[048] Launching: GEMINI_FScot_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 24_gemini_fscot_ctx2t_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_FScot_CTX2t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

fi

# ============================================================
# QWEN25_7B — Qwen/Qwen2.5-7B-Instruct
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "qwen25_7b" ]]; then

mkdir -p Decoder-Only/Qwen2.5/slurm_logs
mkdir -p Decoder-Only/Qwen2.5/data

echo "[049] Launching: QWEN25_7B_ZS_noCTX_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 01_qwen25_7b_zs_noctx_noleg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_ZS_noCTX_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt"

echo "[050] Launching: QWEN25_7B_ZS_noCTX_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 02_qwen25_7b_zs_noctx_leg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_ZS_noCTX_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[051] Launching: QWEN25_7B_ZS_CTX1t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 03_qwen25_7b_zs_ctx1t_noleg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_ZS_CTX1t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[052] Launching: QWEN25_7B_ZS_CTX1t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 04_qwen25_7b_zs_ctx1t_leg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_ZS_CTX1t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[053] Launching: QWEN25_7B_ZS_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 05_qwen25_7b_zs_ctx2t_noleg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_ZS_CTX2t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[054] Launching: QWEN25_7B_ZS_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 06_qwen25_7b_zs_ctx2t_leg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_ZS_CTX2t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[055] Launching: QWEN25_7B_FS_noCTX_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 07_qwen25_7b_fs_noctx_noleg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_FS_noCTX_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt"

echo "[056] Launching: QWEN25_7B_FS_noCTX_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 08_qwen25_7b_fs_noctx_leg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_FS_noCTX_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[057] Launching: QWEN25_7B_FS_CTX1t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 09_qwen25_7b_fs_ctx1t_noleg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_FS_CTX1t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[058] Launching: QWEN25_7B_FS_CTX1t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 10_qwen25_7b_fs_ctx1t_leg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_FS_CTX1t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[059] Launching: QWEN25_7B_FS_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 11_qwen25_7b_fs_ctx2t_noleg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_FS_CTX2t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[060] Launching: QWEN25_7B_FS_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 12_qwen25_7b_fs_ctx2t_leg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_FS_CTX2t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[061] Launching: QWEN25_7B_ZScot_noCTX_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 13_qwen25_7b_zscot_noctx_noleg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_ZScot_noCTX_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt"

echo "[062] Launching: QWEN25_7B_ZScot_noCTX_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 14_qwen25_7b_zscot_noctx_leg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_ZScot_noCTX_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[063] Launching: QWEN25_7B_ZScot_CTX1t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 15_qwen25_7b_zscot_ctx1t_noleg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_ZScot_CTX1t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[064] Launching: QWEN25_7B_ZScot_CTX1t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 16_qwen25_7b_zscot_ctx1t_leg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_ZScot_CTX1t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[065] Launching: QWEN25_7B_ZScot_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 17_qwen25_7b_zscot_ctx2t_noleg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_ZScot_CTX2t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[066] Launching: QWEN25_7B_ZScot_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 18_qwen25_7b_zscot_ctx2t_leg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_ZScot_CTX2t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[067] Launching: QWEN25_7B_FScot_noCTX_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 19_qwen25_7b_fscot_noctx_noleg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_FScot_noCTX_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt"

echo "[068] Launching: QWEN25_7B_FScot_noCTX_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 20_qwen25_7b_fscot_noctx_leg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_FScot_noCTX_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[069] Launching: QWEN25_7B_FScot_CTX1t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 21_qwen25_7b_fscot_ctx1t_noleg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_FScot_CTX1t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[070] Launching: QWEN25_7B_FScot_CTX1t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 22_qwen25_7b_fscot_ctx1t_leg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_FScot_CTX1t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[071] Launching: QWEN25_7B_FScot_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 23_qwen25_7b_fscot_ctx2t_noleg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_FScot_CTX2t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[072] Launching: QWEN25_7B_FScot_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 24_qwen25_7b_fscot_ctx2t_leg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_FScot_CTX2t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

fi

# ============================================================
# GEMINI3_PRO — gemini-3-pro-preview
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "gemini3_pro" ]]; then

mkdir -p Decoder-Only/Gemini3_Pro/slurm_logs
mkdir -p Decoder-Only/Gemini3_Pro/data

echo "[073] Launching: GEMINI3_PRO_ZS_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 01_gemini3_pro_zs_noctx_noleg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-3-pro-preview \
    --backend gemini3 \
    --sheet GEMINI3_PRO_ZS_noCTX_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --thinking_level high"

echo "[074] Launching: GEMINI3_PRO_ZS_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 02_gemini3_pro_zs_noctx_leg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-3-pro-preview \
    --backend gemini3 \
    --sheet GEMINI3_PRO_ZS_noCTX_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --dialect_legitimacy \
    --thinking_level high"

echo "[075] Launching: GEMINI3_PRO_ZS_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 03_gemini3_pro_zs_ctx1t_noleg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-3-pro-preview \
    --backend gemini3 \
    --sheet GEMINI3_PRO_ZS_CTX1t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --thinking_level high"

echo "[076] Launching: GEMINI3_PRO_ZS_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 04_gemini3_pro_zs_ctx1t_leg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-3-pro-preview \
    --backend gemini3 \
    --sheet GEMINI3_PRO_ZS_CTX1t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy \
    --thinking_level high"

echo "[077] Launching: GEMINI3_PRO_ZS_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 05_gemini3_pro_zs_ctx2t_noleg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-3-pro-preview \
    --backend gemini3 \
    --sheet GEMINI3_PRO_ZS_CTX2t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --thinking_level high"

echo "[078] Launching: GEMINI3_PRO_ZS_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 06_gemini3_pro_zs_ctx2t_leg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-3-pro-preview \
    --backend gemini3 \
    --sheet GEMINI3_PRO_ZS_CTX2t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy \
    --thinking_level high"

echo "[079] Launching: GEMINI3_PRO_FS_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 07_gemini3_pro_fs_noctx_noleg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-3-pro-preview \
    --backend gemini3 \
    --sheet GEMINI3_PRO_FS_noCTX_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --thinking_level high"

echo "[080] Launching: GEMINI3_PRO_FS_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 08_gemini3_pro_fs_noctx_leg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-3-pro-preview \
    --backend gemini3 \
    --sheet GEMINI3_PRO_FS_noCTX_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --dialect_legitimacy \
    --thinking_level high"

echo "[081] Launching: GEMINI3_PRO_FS_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 09_gemini3_pro_fs_ctx1t_noleg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-3-pro-preview \
    --backend gemini3 \
    --sheet GEMINI3_PRO_FS_CTX1t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --thinking_level high"

echo "[082] Launching: GEMINI3_PRO_FS_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 10_gemini3_pro_fs_ctx1t_leg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-3-pro-preview \
    --backend gemini3 \
    --sheet GEMINI3_PRO_FS_CTX1t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy \
    --thinking_level high"

echo "[083] Launching: GEMINI3_PRO_FS_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 11_gemini3_pro_fs_ctx2t_noleg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-3-pro-preview \
    --backend gemini3 \
    --sheet GEMINI3_PRO_FS_CTX2t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --thinking_level high"

echo "[084] Launching: GEMINI3_PRO_FS_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 12_gemini3_pro_fs_ctx2t_leg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-3-pro-preview \
    --backend gemini3 \
    --sheet GEMINI3_PRO_FS_CTX2t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy \
    --thinking_level high"

echo "[085] Launching: GEMINI3_PRO_ZScot_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 13_gemini3_pro_zscot_noctx_noleg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-3-pro-preview \
    --backend gemini3 \
    --sheet GEMINI3_PRO_ZScot_noCTX_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --thinking_level high"

echo "[086] Launching: GEMINI3_PRO_ZScot_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 14_gemini3_pro_zscot_noctx_leg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-3-pro-preview \
    --backend gemini3 \
    --sheet GEMINI3_PRO_ZScot_noCTX_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --dialect_legitimacy \
    --thinking_level high"

echo "[087] Launching: GEMINI3_PRO_ZScot_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 15_gemini3_pro_zscot_ctx1t_noleg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-3-pro-preview \
    --backend gemini3 \
    --sheet GEMINI3_PRO_ZScot_CTX1t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --thinking_level high"

echo "[088] Launching: GEMINI3_PRO_ZScot_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 16_gemini3_pro_zscot_ctx1t_leg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-3-pro-preview \
    --backend gemini3 \
    --sheet GEMINI3_PRO_ZScot_CTX1t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy \
    --thinking_level high"

echo "[089] Launching: GEMINI3_PRO_ZScot_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 17_gemini3_pro_zscot_ctx2t_noleg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-3-pro-preview \
    --backend gemini3 \
    --sheet GEMINI3_PRO_ZScot_CTX2t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --thinking_level high"

echo "[090] Launching: GEMINI3_PRO_ZScot_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 18_gemini3_pro_zscot_ctx2t_leg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-3-pro-preview \
    --backend gemini3 \
    --sheet GEMINI3_PRO_ZScot_CTX2t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy \
    --thinking_level high"

echo "[091] Launching: GEMINI3_PRO_FScot_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 19_gemini3_pro_fscot_noctx_noleg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-3-pro-preview \
    --backend gemini3 \
    --sheet GEMINI3_PRO_FScot_noCTX_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --thinking_level high"

echo "[092] Launching: GEMINI3_PRO_FScot_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 20_gemini3_pro_fscot_noctx_leg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-3-pro-preview \
    --backend gemini3 \
    --sheet GEMINI3_PRO_FScot_noCTX_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --dialect_legitimacy \
    --thinking_level high"

echo "[093] Launching: GEMINI3_PRO_FScot_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 21_gemini3_pro_fscot_ctx1t_noleg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-3-pro-preview \
    --backend gemini3 \
    --sheet GEMINI3_PRO_FScot_CTX1t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --thinking_level high"

echo "[094] Launching: GEMINI3_PRO_FScot_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 22_gemini3_pro_fscot_ctx1t_leg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-3-pro-preview \
    --backend gemini3 \
    --sheet GEMINI3_PRO_FScot_CTX1t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy \
    --thinking_level high"

echo "[095] Launching: GEMINI3_PRO_FScot_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 23_gemini3_pro_fscot_ctx2t_noleg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-3-pro-preview \
    --backend gemini3 \
    --sheet GEMINI3_PRO_FScot_CTX2t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --thinking_level high"

echo "[096] Launching: GEMINI3_PRO_FScot_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 24_gemini3_pro_fscot_ctx2t_leg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gemini-3-pro-preview \
    --backend gemini3 \
    --sheet GEMINI3_PRO_FScot_CTX2t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy \
    --thinking_level high"

fi

# ============================================================
# GPT41 — gpt-4.1
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "gpt41" ]]; then

mkdir -p Decoder-Only/GPT41/slurm_logs
mkdir -p Decoder-Only/GPT41/data

echo "[097] Launching: GPT41_ZS_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 01_gpt41_zs_noctx_noleg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_ZS_noCTX_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt"

echo "[098] Launching: GPT41_ZS_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 02_gpt41_zs_noctx_leg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_ZS_noCTX_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[099] Launching: GPT41_ZS_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 03_gpt41_zs_ctx1t_noleg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_ZS_CTX1t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[100] Launching: GPT41_ZS_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 04_gpt41_zs_ctx1t_leg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_ZS_CTX1t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[101] Launching: GPT41_ZS_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 05_gpt41_zs_ctx2t_noleg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_ZS_CTX2t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[102] Launching: GPT41_ZS_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 06_gpt41_zs_ctx2t_leg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_ZS_CTX2t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[103] Launching: GPT41_FS_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 07_gpt41_fs_noctx_noleg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_FS_noCTX_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt"

echo "[104] Launching: GPT41_FS_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 08_gpt41_fs_noctx_leg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_FS_noCTX_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[105] Launching: GPT41_FS_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 09_gpt41_fs_ctx1t_noleg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_FS_CTX1t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[106] Launching: GPT41_FS_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 10_gpt41_fs_ctx1t_leg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_FS_CTX1t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[107] Launching: GPT41_FS_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 11_gpt41_fs_ctx2t_noleg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_FS_CTX2t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[108] Launching: GPT41_FS_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 12_gpt41_fs_ctx2t_leg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_FS_CTX2t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[109] Launching: GPT41_ZScot_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 13_gpt41_zscot_noctx_noleg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_ZScot_noCTX_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt"

echo "[110] Launching: GPT41_ZScot_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 14_gpt41_zscot_noctx_leg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_ZScot_noCTX_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[111] Launching: GPT41_ZScot_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 15_gpt41_zscot_ctx1t_noleg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_ZScot_CTX1t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[112] Launching: GPT41_ZScot_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 16_gpt41_zscot_ctx1t_leg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_ZScot_CTX1t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[113] Launching: GPT41_ZScot_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 17_gpt41_zscot_ctx2t_noleg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_ZScot_CTX2t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[114] Launching: GPT41_ZScot_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 18_gpt41_zscot_ctx2t_leg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_ZScot_CTX2t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[115] Launching: GPT41_FScot_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 19_gpt41_fscot_noctx_noleg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_FScot_noCTX_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt"

echo "[116] Launching: GPT41_FScot_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 20_gpt41_fscot_noctx_leg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_FScot_noCTX_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[117] Launching: GPT41_FScot_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 21_gpt41_fscot_ctx1t_noleg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_FScot_CTX1t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[118] Launching: GPT41_FScot_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 22_gpt41_fscot_ctx1t_leg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_FScot_CTX1t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[119] Launching: GPT41_FScot_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 23_gpt41_fscot_ctx2t_noleg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_FScot_CTX2t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[120] Launching: GPT41_FScot_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 24_gpt41_fscot_ctx2t_leg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_FScot_CTX2t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

fi

# ============================================================
# GPT52_INSTANT — gpt-5.2-chat-latest
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "gpt52_instant" ]]; then

mkdir -p Decoder-Only/GPT52_Instant/slurm_logs
mkdir -p Decoder-Only/GPT52_Instant/data

echo "[121] Launching: GPT52_INSTANT_ZS_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 01_gpt52_instant_zs_noctx_noleg \
  -o Decoder-Only/GPT52_Instant/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2-chat-latest \
    --backend openai \
    --sheet GPT52_INSTANT_ZS_noCTX_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Instant/data \
    --dump_prompt"

echo "[122] Launching: GPT52_INSTANT_ZS_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 02_gpt52_instant_zs_noctx_leg \
  -o Decoder-Only/GPT52_Instant/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2-chat-latest \
    --backend openai \
    --sheet GPT52_INSTANT_ZS_noCTX_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Instant/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[123] Launching: GPT52_INSTANT_ZS_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 03_gpt52_instant_zs_ctx1t_noleg \
  -o Decoder-Only/GPT52_Instant/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2-chat-latest \
    --backend openai \
    --sheet GPT52_INSTANT_ZS_CTX1t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Instant/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[124] Launching: GPT52_INSTANT_ZS_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 04_gpt52_instant_zs_ctx1t_leg \
  -o Decoder-Only/GPT52_Instant/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2-chat-latest \
    --backend openai \
    --sheet GPT52_INSTANT_ZS_CTX1t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Instant/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[125] Launching: GPT52_INSTANT_ZS_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 05_gpt52_instant_zs_ctx2t_noleg \
  -o Decoder-Only/GPT52_Instant/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2-chat-latest \
    --backend openai \
    --sheet GPT52_INSTANT_ZS_CTX2t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Instant/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[126] Launching: GPT52_INSTANT_ZS_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 06_gpt52_instant_zs_ctx2t_leg \
  -o Decoder-Only/GPT52_Instant/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2-chat-latest \
    --backend openai \
    --sheet GPT52_INSTANT_ZS_CTX2t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Instant/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[127] Launching: GPT52_INSTANT_FS_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 07_gpt52_instant_fs_noctx_noleg \
  -o Decoder-Only/GPT52_Instant/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2-chat-latest \
    --backend openai \
    --sheet GPT52_INSTANT_FS_noCTX_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Instant/data \
    --dump_prompt"

echo "[128] Launching: GPT52_INSTANT_FS_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 08_gpt52_instant_fs_noctx_leg \
  -o Decoder-Only/GPT52_Instant/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2-chat-latest \
    --backend openai \
    --sheet GPT52_INSTANT_FS_noCTX_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Instant/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[129] Launching: GPT52_INSTANT_FS_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 09_gpt52_instant_fs_ctx1t_noleg \
  -o Decoder-Only/GPT52_Instant/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2-chat-latest \
    --backend openai \
    --sheet GPT52_INSTANT_FS_CTX1t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Instant/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[130] Launching: GPT52_INSTANT_FS_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 10_gpt52_instant_fs_ctx1t_leg \
  -o Decoder-Only/GPT52_Instant/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2-chat-latest \
    --backend openai \
    --sheet GPT52_INSTANT_FS_CTX1t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Instant/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[131] Launching: GPT52_INSTANT_FS_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 11_gpt52_instant_fs_ctx2t_noleg \
  -o Decoder-Only/GPT52_Instant/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2-chat-latest \
    --backend openai \
    --sheet GPT52_INSTANT_FS_CTX2t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Instant/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[132] Launching: GPT52_INSTANT_FS_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 12_gpt52_instant_fs_ctx2t_leg \
  -o Decoder-Only/GPT52_Instant/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2-chat-latest \
    --backend openai \
    --sheet GPT52_INSTANT_FS_CTX2t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Instant/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[133] Launching: GPT52_INSTANT_ZScot_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 13_gpt52_instant_zscot_noctx_noleg \
  -o Decoder-Only/GPT52_Instant/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2-chat-latest \
    --backend openai \
    --sheet GPT52_INSTANT_ZScot_noCTX_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Instant/data \
    --dump_prompt"

echo "[134] Launching: GPT52_INSTANT_ZScot_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 14_gpt52_instant_zscot_noctx_leg \
  -o Decoder-Only/GPT52_Instant/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2-chat-latest \
    --backend openai \
    --sheet GPT52_INSTANT_ZScot_noCTX_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Instant/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[135] Launching: GPT52_INSTANT_ZScot_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 15_gpt52_instant_zscot_ctx1t_noleg \
  -o Decoder-Only/GPT52_Instant/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2-chat-latest \
    --backend openai \
    --sheet GPT52_INSTANT_ZScot_CTX1t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Instant/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[136] Launching: GPT52_INSTANT_ZScot_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 16_gpt52_instant_zscot_ctx1t_leg \
  -o Decoder-Only/GPT52_Instant/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2-chat-latest \
    --backend openai \
    --sheet GPT52_INSTANT_ZScot_CTX1t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Instant/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[137] Launching: GPT52_INSTANT_ZScot_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 17_gpt52_instant_zscot_ctx2t_noleg \
  -o Decoder-Only/GPT52_Instant/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2-chat-latest \
    --backend openai \
    --sheet GPT52_INSTANT_ZScot_CTX2t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Instant/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[138] Launching: GPT52_INSTANT_ZScot_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 18_gpt52_instant_zscot_ctx2t_leg \
  -o Decoder-Only/GPT52_Instant/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2-chat-latest \
    --backend openai \
    --sheet GPT52_INSTANT_ZScot_CTX2t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Instant/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[139] Launching: GPT52_INSTANT_FScot_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 19_gpt52_instant_fscot_noctx_noleg \
  -o Decoder-Only/GPT52_Instant/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2-chat-latest \
    --backend openai \
    --sheet GPT52_INSTANT_FScot_noCTX_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Instant/data \
    --dump_prompt"

echo "[140] Launching: GPT52_INSTANT_FScot_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 20_gpt52_instant_fscot_noctx_leg \
  -o Decoder-Only/GPT52_Instant/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2-chat-latest \
    --backend openai \
    --sheet GPT52_INSTANT_FScot_noCTX_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Instant/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[141] Launching: GPT52_INSTANT_FScot_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 21_gpt52_instant_fscot_ctx1t_noleg \
  -o Decoder-Only/GPT52_Instant/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2-chat-latest \
    --backend openai \
    --sheet GPT52_INSTANT_FScot_CTX1t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Instant/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[142] Launching: GPT52_INSTANT_FScot_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 22_gpt52_instant_fscot_ctx1t_leg \
  -o Decoder-Only/GPT52_Instant/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2-chat-latest \
    --backend openai \
    --sheet GPT52_INSTANT_FScot_CTX1t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Instant/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[143] Launching: GPT52_INSTANT_FScot_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 23_gpt52_instant_fscot_ctx2t_noleg \
  -o Decoder-Only/GPT52_Instant/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2-chat-latest \
    --backend openai \
    --sheet GPT52_INSTANT_FScot_CTX2t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Instant/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[144] Launching: GPT52_INSTANT_FScot_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 24_gpt52_instant_fscot_ctx2t_leg \
  -o Decoder-Only/GPT52_Instant/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2-chat-latest \
    --backend openai \
    --sheet GPT52_INSTANT_FScot_CTX2t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Instant/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

fi

# ============================================================
# GPT52_THINK_MED — gpt-5.2
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "gpt52_think_med" ]]; then

mkdir -p Decoder-Only/GPT52_Thinking_Med/slurm_logs
mkdir -p Decoder-Only/GPT52_Thinking_Med/data

echo "[145] Launching: GPT52_THINK_MED_ZS_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 01_gpt52_think_med_zs_noctx_noleg \
  -o Decoder-Only/GPT52_Thinking_Med/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_MED_ZS_noCTX_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_Med/data \
    --dump_prompt \
    --reasoning_effort medium"

echo "[146] Launching: GPT52_THINK_MED_ZS_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 02_gpt52_think_med_zs_noctx_leg \
  -o Decoder-Only/GPT52_Thinking_Med/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_MED_ZS_noCTX_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_Med/data \
    --dump_prompt \
    --dialect_legitimacy \
    --reasoning_effort medium"

echo "[147] Launching: GPT52_THINK_MED_ZS_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 03_gpt52_think_med_zs_ctx1t_noleg \
  -o Decoder-Only/GPT52_Thinking_Med/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_MED_ZS_CTX1t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_Med/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --reasoning_effort medium"

echo "[148] Launching: GPT52_THINK_MED_ZS_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 04_gpt52_think_med_zs_ctx1t_leg \
  -o Decoder-Only/GPT52_Thinking_Med/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_MED_ZS_CTX1t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_Med/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy \
    --reasoning_effort medium"

echo "[149] Launching: GPT52_THINK_MED_ZS_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 05_gpt52_think_med_zs_ctx2t_noleg \
  -o Decoder-Only/GPT52_Thinking_Med/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_MED_ZS_CTX2t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_Med/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --reasoning_effort medium"

echo "[150] Launching: GPT52_THINK_MED_ZS_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 06_gpt52_think_med_zs_ctx2t_leg \
  -o Decoder-Only/GPT52_Thinking_Med/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_MED_ZS_CTX2t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_Med/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy \
    --reasoning_effort medium"

echo "[151] Launching: GPT52_THINK_MED_FS_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 07_gpt52_think_med_fs_noctx_noleg \
  -o Decoder-Only/GPT52_Thinking_Med/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_MED_FS_noCTX_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_Med/data \
    --dump_prompt \
    --reasoning_effort medium"

echo "[152] Launching: GPT52_THINK_MED_FS_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 08_gpt52_think_med_fs_noctx_leg \
  -o Decoder-Only/GPT52_Thinking_Med/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_MED_FS_noCTX_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_Med/data \
    --dump_prompt \
    --dialect_legitimacy \
    --reasoning_effort medium"

echo "[153] Launching: GPT52_THINK_MED_FS_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 09_gpt52_think_med_fs_ctx1t_noleg \
  -o Decoder-Only/GPT52_Thinking_Med/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_MED_FS_CTX1t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_Med/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --reasoning_effort medium"

echo "[154] Launching: GPT52_THINK_MED_FS_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 10_gpt52_think_med_fs_ctx1t_leg \
  -o Decoder-Only/GPT52_Thinking_Med/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_MED_FS_CTX1t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_Med/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy \
    --reasoning_effort medium"

echo "[155] Launching: GPT52_THINK_MED_FS_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 11_gpt52_think_med_fs_ctx2t_noleg \
  -o Decoder-Only/GPT52_Thinking_Med/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_MED_FS_CTX2t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_Med/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --reasoning_effort medium"

echo "[156] Launching: GPT52_THINK_MED_FS_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 12_gpt52_think_med_fs_ctx2t_leg \
  -o Decoder-Only/GPT52_Thinking_Med/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_MED_FS_CTX2t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_Med/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy \
    --reasoning_effort medium"

echo "[157] Launching: GPT52_THINK_MED_ZScot_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 13_gpt52_think_med_zscot_noctx_noleg \
  -o Decoder-Only/GPT52_Thinking_Med/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_MED_ZScot_noCTX_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_Med/data \
    --dump_prompt \
    --reasoning_effort medium"

echo "[158] Launching: GPT52_THINK_MED_ZScot_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 14_gpt52_think_med_zscot_noctx_leg \
  -o Decoder-Only/GPT52_Thinking_Med/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_MED_ZScot_noCTX_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_Med/data \
    --dump_prompt \
    --dialect_legitimacy \
    --reasoning_effort medium"

echo "[159] Launching: GPT52_THINK_MED_ZScot_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 15_gpt52_think_med_zscot_ctx1t_noleg \
  -o Decoder-Only/GPT52_Thinking_Med/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_MED_ZScot_CTX1t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_Med/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --reasoning_effort medium"

echo "[160] Launching: GPT52_THINK_MED_ZScot_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 16_gpt52_think_med_zscot_ctx1t_leg \
  -o Decoder-Only/GPT52_Thinking_Med/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_MED_ZScot_CTX1t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_Med/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy \
    --reasoning_effort medium"

echo "[161] Launching: GPT52_THINK_MED_ZScot_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 17_gpt52_think_med_zscot_ctx2t_noleg \
  -o Decoder-Only/GPT52_Thinking_Med/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_MED_ZScot_CTX2t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_Med/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --reasoning_effort medium"

echo "[162] Launching: GPT52_THINK_MED_ZScot_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 18_gpt52_think_med_zscot_ctx2t_leg \
  -o Decoder-Only/GPT52_Thinking_Med/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_MED_ZScot_CTX2t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_Med/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy \
    --reasoning_effort medium"

echo "[163] Launching: GPT52_THINK_MED_FScot_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 19_gpt52_think_med_fscot_noctx_noleg \
  -o Decoder-Only/GPT52_Thinking_Med/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_MED_FScot_noCTX_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_Med/data \
    --dump_prompt \
    --reasoning_effort medium"

echo "[164] Launching: GPT52_THINK_MED_FScot_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 20_gpt52_think_med_fscot_noctx_leg \
  -o Decoder-Only/GPT52_Thinking_Med/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_MED_FScot_noCTX_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_Med/data \
    --dump_prompt \
    --dialect_legitimacy \
    --reasoning_effort medium"

echo "[165] Launching: GPT52_THINK_MED_FScot_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 21_gpt52_think_med_fscot_ctx1t_noleg \
  -o Decoder-Only/GPT52_Thinking_Med/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_MED_FScot_CTX1t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_Med/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --reasoning_effort medium"

echo "[166] Launching: GPT52_THINK_MED_FScot_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 22_gpt52_think_med_fscot_ctx1t_leg \
  -o Decoder-Only/GPT52_Thinking_Med/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_MED_FScot_CTX1t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_Med/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy \
    --reasoning_effort medium"

echo "[167] Launching: GPT52_THINK_MED_FScot_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 23_gpt52_think_med_fscot_ctx2t_noleg \
  -o Decoder-Only/GPT52_Thinking_Med/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_MED_FScot_CTX2t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_Med/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --reasoning_effort medium"

echo "[168] Launching: GPT52_THINK_MED_FScot_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 24_gpt52_think_med_fscot_ctx2t_leg \
  -o Decoder-Only/GPT52_Thinking_Med/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_MED_FScot_CTX2t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_Med/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy \
    --reasoning_effort medium"

fi

# ============================================================
# GPT52_THINK_HIGH — gpt-5.2
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "gpt52_think_high" ]]; then

mkdir -p Decoder-Only/GPT52_Thinking_High/slurm_logs
mkdir -p Decoder-Only/GPT52_Thinking_High/data

echo "[169] Launching: GPT52_THINK_HIGH_ZS_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 01_gpt52_think_high_zs_noctx_noleg \
  -o Decoder-Only/GPT52_Thinking_High/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_HIGH_ZS_noCTX_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_High/data \
    --dump_prompt \
    --reasoning_effort high"

echo "[170] Launching: GPT52_THINK_HIGH_ZS_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 02_gpt52_think_high_zs_noctx_leg \
  -o Decoder-Only/GPT52_Thinking_High/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_HIGH_ZS_noCTX_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_High/data \
    --dump_prompt \
    --dialect_legitimacy \
    --reasoning_effort high"

echo "[171] Launching: GPT52_THINK_HIGH_ZS_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 03_gpt52_think_high_zs_ctx1t_noleg \
  -o Decoder-Only/GPT52_Thinking_High/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_HIGH_ZS_CTX1t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_High/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --reasoning_effort high"

echo "[172] Launching: GPT52_THINK_HIGH_ZS_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 04_gpt52_think_high_zs_ctx1t_leg \
  -o Decoder-Only/GPT52_Thinking_High/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_HIGH_ZS_CTX1t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_High/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy \
    --reasoning_effort high"

echo "[173] Launching: GPT52_THINK_HIGH_ZS_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 05_gpt52_think_high_zs_ctx2t_noleg \
  -o Decoder-Only/GPT52_Thinking_High/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_HIGH_ZS_CTX2t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_High/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --reasoning_effort high"

echo "[174] Launching: GPT52_THINK_HIGH_ZS_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 06_gpt52_think_high_zs_ctx2t_leg \
  -o Decoder-Only/GPT52_Thinking_High/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_HIGH_ZS_CTX2t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_High/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy \
    --reasoning_effort high"

echo "[175] Launching: GPT52_THINK_HIGH_FS_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 07_gpt52_think_high_fs_noctx_noleg \
  -o Decoder-Only/GPT52_Thinking_High/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_HIGH_FS_noCTX_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_High/data \
    --dump_prompt \
    --reasoning_effort high"

echo "[176] Launching: GPT52_THINK_HIGH_FS_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 08_gpt52_think_high_fs_noctx_leg \
  -o Decoder-Only/GPT52_Thinking_High/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_HIGH_FS_noCTX_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_High/data \
    --dump_prompt \
    --dialect_legitimacy \
    --reasoning_effort high"

echo "[177] Launching: GPT52_THINK_HIGH_FS_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 09_gpt52_think_high_fs_ctx1t_noleg \
  -o Decoder-Only/GPT52_Thinking_High/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_HIGH_FS_CTX1t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_High/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --reasoning_effort high"

echo "[178] Launching: GPT52_THINK_HIGH_FS_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 10_gpt52_think_high_fs_ctx1t_leg \
  -o Decoder-Only/GPT52_Thinking_High/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_HIGH_FS_CTX1t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_High/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy \
    --reasoning_effort high"

echo "[179] Launching: GPT52_THINK_HIGH_FS_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 11_gpt52_think_high_fs_ctx2t_noleg \
  -o Decoder-Only/GPT52_Thinking_High/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_HIGH_FS_CTX2t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_High/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --reasoning_effort high"

echo "[180] Launching: GPT52_THINK_HIGH_FS_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 12_gpt52_think_high_fs_ctx2t_leg \
  -o Decoder-Only/GPT52_Thinking_High/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_HIGH_FS_CTX2t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_High/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy \
    --reasoning_effort high"

echo "[181] Launching: GPT52_THINK_HIGH_ZScot_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 13_gpt52_think_high_zscot_noctx_noleg \
  -o Decoder-Only/GPT52_Thinking_High/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_HIGH_ZScot_noCTX_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_High/data \
    --dump_prompt \
    --reasoning_effort high"

echo "[182] Launching: GPT52_THINK_HIGH_ZScot_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 14_gpt52_think_high_zscot_noctx_leg \
  -o Decoder-Only/GPT52_Thinking_High/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_HIGH_ZScot_noCTX_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_High/data \
    --dump_prompt \
    --dialect_legitimacy \
    --reasoning_effort high"

echo "[183] Launching: GPT52_THINK_HIGH_ZScot_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 15_gpt52_think_high_zscot_ctx1t_noleg \
  -o Decoder-Only/GPT52_Thinking_High/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_HIGH_ZScot_CTX1t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_High/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --reasoning_effort high"

echo "[184] Launching: GPT52_THINK_HIGH_ZScot_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 16_gpt52_think_high_zscot_ctx1t_leg \
  -o Decoder-Only/GPT52_Thinking_High/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_HIGH_ZScot_CTX1t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_High/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy \
    --reasoning_effort high"

echo "[185] Launching: GPT52_THINK_HIGH_ZScot_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 17_gpt52_think_high_zscot_ctx2t_noleg \
  -o Decoder-Only/GPT52_Thinking_High/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_HIGH_ZScot_CTX2t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_High/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --reasoning_effort high"

echo "[186] Launching: GPT52_THINK_HIGH_ZScot_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 18_gpt52_think_high_zscot_ctx2t_leg \
  -o Decoder-Only/GPT52_Thinking_High/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_HIGH_ZScot_CTX2t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_High/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy \
    --reasoning_effort high"

echo "[187] Launching: GPT52_THINK_HIGH_FScot_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 19_gpt52_think_high_fscot_noctx_noleg \
  -o Decoder-Only/GPT52_Thinking_High/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_HIGH_FScot_noCTX_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_High/data \
    --dump_prompt \
    --reasoning_effort high"

echo "[188] Launching: GPT52_THINK_HIGH_FScot_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 20_gpt52_think_high_fscot_noctx_leg \
  -o Decoder-Only/GPT52_Thinking_High/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_HIGH_FScot_noCTX_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_High/data \
    --dump_prompt \
    --dialect_legitimacy \
    --reasoning_effort high"

echo "[189] Launching: GPT52_THINK_HIGH_FScot_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 21_gpt52_think_high_fscot_ctx1t_noleg \
  -o Decoder-Only/GPT52_Thinking_High/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_HIGH_FScot_CTX1t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_High/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --reasoning_effort high"

echo "[190] Launching: GPT52_THINK_HIGH_FScot_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 22_gpt52_think_high_fscot_ctx1t_leg \
  -o Decoder-Only/GPT52_Thinking_High/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_HIGH_FScot_CTX1t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_High/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy \
    --reasoning_effort high"

echo "[191] Launching: GPT52_THINK_HIGH_FScot_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 23_gpt52_think_high_fscot_ctx2t_noleg \
  -o Decoder-Only/GPT52_Thinking_High/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_HIGH_FScot_CTX2t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_High/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --reasoning_effort high"

echo "[192] Launching: GPT52_THINK_HIGH_FScot_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n 24_gpt52_think_high_fscot_ctx2t_leg \
  -o Decoder-Only/GPT52_Thinking_High/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model gpt-5.2 \
    --backend openai_reasoning \
    --sheet GPT52_THINK_HIGH_FScot_CTX2t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT52_Thinking_High/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy \
    --reasoning_effort high"

fi

# ============================================================
# PHI4_REASONING — microsoft/Phi-4-reasoning
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "phi4_reasoning" ]]; then

mkdir -p Decoder-Only/Phi-4-reasoning/slurm_logs
mkdir -p Decoder-Only/Phi-4-reasoning/data

echo "[193] Launching: PHI4_REASONING_ZS_noCTX_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 01_phi4_reasoning_zs_noctx_noleg \
  -o Decoder-Only/Phi-4-reasoning/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/Phi-4-reasoning \
    --backend phi_reasoning \
    --sheet PHI4_REASONING_ZS_noCTX_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4-reasoning/data \
    --dump_prompt"

echo "[194] Launching: PHI4_REASONING_ZS_noCTX_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 02_phi4_reasoning_zs_noctx_leg \
  -o Decoder-Only/Phi-4-reasoning/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/Phi-4-reasoning \
    --backend phi_reasoning \
    --sheet PHI4_REASONING_ZS_noCTX_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4-reasoning/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[195] Launching: PHI4_REASONING_ZS_CTX1t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 03_phi4_reasoning_zs_ctx1t_noleg \
  -o Decoder-Only/Phi-4-reasoning/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/Phi-4-reasoning \
    --backend phi_reasoning \
    --sheet PHI4_REASONING_ZS_CTX1t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4-reasoning/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[196] Launching: PHI4_REASONING_ZS_CTX1t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 04_phi4_reasoning_zs_ctx1t_leg \
  -o Decoder-Only/Phi-4-reasoning/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/Phi-4-reasoning \
    --backend phi_reasoning \
    --sheet PHI4_REASONING_ZS_CTX1t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4-reasoning/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[197] Launching: PHI4_REASONING_ZS_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 05_phi4_reasoning_zs_ctx2t_noleg \
  -o Decoder-Only/Phi-4-reasoning/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/Phi-4-reasoning \
    --backend phi_reasoning \
    --sheet PHI4_REASONING_ZS_CTX2t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4-reasoning/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[198] Launching: PHI4_REASONING_ZS_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 06_phi4_reasoning_zs_ctx2t_leg \
  -o Decoder-Only/Phi-4-reasoning/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/Phi-4-reasoning \
    --backend phi_reasoning \
    --sheet PHI4_REASONING_ZS_CTX2t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4-reasoning/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[199] Launching: PHI4_REASONING_FS_noCTX_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 07_phi4_reasoning_fs_noctx_noleg \
  -o Decoder-Only/Phi-4-reasoning/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/Phi-4-reasoning \
    --backend phi_reasoning \
    --sheet PHI4_REASONING_FS_noCTX_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4-reasoning/data \
    --dump_prompt"

echo "[200] Launching: PHI4_REASONING_FS_noCTX_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 08_phi4_reasoning_fs_noctx_leg \
  -o Decoder-Only/Phi-4-reasoning/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/Phi-4-reasoning \
    --backend phi_reasoning \
    --sheet PHI4_REASONING_FS_noCTX_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4-reasoning/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[201] Launching: PHI4_REASONING_FS_CTX1t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 09_phi4_reasoning_fs_ctx1t_noleg \
  -o Decoder-Only/Phi-4-reasoning/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/Phi-4-reasoning \
    --backend phi_reasoning \
    --sheet PHI4_REASONING_FS_CTX1t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4-reasoning/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[202] Launching: PHI4_REASONING_FS_CTX1t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 10_phi4_reasoning_fs_ctx1t_leg \
  -o Decoder-Only/Phi-4-reasoning/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/Phi-4-reasoning \
    --backend phi_reasoning \
    --sheet PHI4_REASONING_FS_CTX1t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4-reasoning/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[203] Launching: PHI4_REASONING_FS_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 11_phi4_reasoning_fs_ctx2t_noleg \
  -o Decoder-Only/Phi-4-reasoning/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/Phi-4-reasoning \
    --backend phi_reasoning \
    --sheet PHI4_REASONING_FS_CTX2t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4-reasoning/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[204] Launching: PHI4_REASONING_FS_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 12_phi4_reasoning_fs_ctx2t_leg \
  -o Decoder-Only/Phi-4-reasoning/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/Phi-4-reasoning \
    --backend phi_reasoning \
    --sheet PHI4_REASONING_FS_CTX2t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4-reasoning/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[205] Launching: PHI4_REASONING_ZScot_noCTX_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 13_phi4_reasoning_zscot_noctx_noleg \
  -o Decoder-Only/Phi-4-reasoning/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/Phi-4-reasoning \
    --backend phi_reasoning \
    --sheet PHI4_REASONING_ZScot_noCTX_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4-reasoning/data \
    --dump_prompt"

echo "[206] Launching: PHI4_REASONING_ZScot_noCTX_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 14_phi4_reasoning_zscot_noctx_leg \
  -o Decoder-Only/Phi-4-reasoning/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/Phi-4-reasoning \
    --backend phi_reasoning \
    --sheet PHI4_REASONING_ZScot_noCTX_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4-reasoning/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[207] Launching: PHI4_REASONING_ZScot_CTX1t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 15_phi4_reasoning_zscot_ctx1t_noleg \
  -o Decoder-Only/Phi-4-reasoning/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/Phi-4-reasoning \
    --backend phi_reasoning \
    --sheet PHI4_REASONING_ZScot_CTX1t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4-reasoning/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[208] Launching: PHI4_REASONING_ZScot_CTX1t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 16_phi4_reasoning_zscot_ctx1t_leg \
  -o Decoder-Only/Phi-4-reasoning/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/Phi-4-reasoning \
    --backend phi_reasoning \
    --sheet PHI4_REASONING_ZScot_CTX1t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4-reasoning/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[209] Launching: PHI4_REASONING_ZScot_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 17_phi4_reasoning_zscot_ctx2t_noleg \
  -o Decoder-Only/Phi-4-reasoning/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/Phi-4-reasoning \
    --backend phi_reasoning \
    --sheet PHI4_REASONING_ZScot_CTX2t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4-reasoning/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[210] Launching: PHI4_REASONING_ZScot_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 18_phi4_reasoning_zscot_ctx2t_leg \
  -o Decoder-Only/Phi-4-reasoning/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/Phi-4-reasoning \
    --backend phi_reasoning \
    --sheet PHI4_REASONING_ZScot_CTX2t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4-reasoning/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[211] Launching: PHI4_REASONING_FScot_noCTX_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 19_phi4_reasoning_fscot_noctx_noleg \
  -o Decoder-Only/Phi-4-reasoning/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/Phi-4-reasoning \
    --backend phi_reasoning \
    --sheet PHI4_REASONING_FScot_noCTX_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4-reasoning/data \
    --dump_prompt"

echo "[212] Launching: PHI4_REASONING_FScot_noCTX_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 20_phi4_reasoning_fscot_noctx_leg \
  -o Decoder-Only/Phi-4-reasoning/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/Phi-4-reasoning \
    --backend phi_reasoning \
    --sheet PHI4_REASONING_FScot_noCTX_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4-reasoning/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[213] Launching: PHI4_REASONING_FScot_CTX1t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 21_phi4_reasoning_fscot_ctx1t_noleg \
  -o Decoder-Only/Phi-4-reasoning/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/Phi-4-reasoning \
    --backend phi_reasoning \
    --sheet PHI4_REASONING_FScot_CTX1t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4-reasoning/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[214] Launching: PHI4_REASONING_FScot_CTX1t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 22_phi4_reasoning_fscot_ctx1t_leg \
  -o Decoder-Only/Phi-4-reasoning/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/Phi-4-reasoning \
    --backend phi_reasoning \
    --sheet PHI4_REASONING_FScot_CTX1t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4-reasoning/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[215] Launching: PHI4_REASONING_FScot_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 23_phi4_reasoning_fscot_ctx2t_noleg \
  -o Decoder-Only/Phi-4-reasoning/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/Phi-4-reasoning \
    --backend phi_reasoning \
    --sheet PHI4_REASONING_FScot_CTX2t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4-reasoning/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[216] Launching: PHI4_REASONING_FScot_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n 24_phi4_reasoning_fscot_ctx2t_leg \
  -o Decoder-Only/Phi-4-reasoning/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model microsoft/Phi-4-reasoning \
    --backend phi_reasoning \
    --sheet PHI4_REASONING_FScot_CTX2t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4-reasoning/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

fi

# ============================================================
# LLAMA70B — meta-llama/Llama-3.1-70B-Instruct
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "llama70b" ]]; then

mkdir -p Decoder-Only/Llama-3.1-70B/slurm_logs
mkdir -p Decoder-Only/Llama-3.1-70B/data

echo "[217] Launching: LLAMA70B_ZS_noCTX_noLeg"
nlprun -g 4 -q sphinx -p standard -r 300G -c 8 \
  -n 01_llama70b_zs_noctx_noleg \
  -o Decoder-Only/Llama-3.1-70B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --backend llama \
    --sheet LLAMA70B_ZS_noCTX_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Llama-3.1-70B/data \
    --dump_prompt"

echo "[218] Launching: LLAMA70B_ZS_noCTX_Leg"
nlprun -g 4 -q sphinx -p standard -r 300G -c 8 \
  -n 02_llama70b_zs_noctx_leg \
  -o Decoder-Only/Llama-3.1-70B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --backend llama \
    --sheet LLAMA70B_ZS_noCTX_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Llama-3.1-70B/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[219] Launching: LLAMA70B_ZS_CTX1t_noLeg"
nlprun -g 4 -q sphinx -p standard -r 300G -c 8 \
  -n 03_llama70b_zs_ctx1t_noleg \
  -o Decoder-Only/Llama-3.1-70B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --backend llama \
    --sheet LLAMA70B_ZS_CTX1t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Llama-3.1-70B/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[220] Launching: LLAMA70B_ZS_CTX1t_Leg"
nlprun -g 4 -q sphinx -p standard -r 300G -c 8 \
  -n 04_llama70b_zs_ctx1t_leg \
  -o Decoder-Only/Llama-3.1-70B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --backend llama \
    --sheet LLAMA70B_ZS_CTX1t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Llama-3.1-70B/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[221] Launching: LLAMA70B_ZS_CTX2t_noLeg"
nlprun -g 4 -q sphinx -p standard -r 300G -c 8 \
  -n 05_llama70b_zs_ctx2t_noleg \
  -o Decoder-Only/Llama-3.1-70B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --backend llama \
    --sheet LLAMA70B_ZS_CTX2t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Llama-3.1-70B/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[222] Launching: LLAMA70B_ZS_CTX2t_Leg"
nlprun -g 4 -q sphinx -p standard -r 300G -c 8 \
  -n 06_llama70b_zs_ctx2t_leg \
  -o Decoder-Only/Llama-3.1-70B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --backend llama \
    --sheet LLAMA70B_ZS_CTX2t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Llama-3.1-70B/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[223] Launching: LLAMA70B_FS_noCTX_noLeg"
nlprun -g 4 -q sphinx -p standard -r 300G -c 8 \
  -n 07_llama70b_fs_noctx_noleg \
  -o Decoder-Only/Llama-3.1-70B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --backend llama \
    --sheet LLAMA70B_FS_noCTX_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Llama-3.1-70B/data \
    --dump_prompt"

echo "[224] Launching: LLAMA70B_FS_noCTX_Leg"
nlprun -g 4 -q sphinx -p standard -r 300G -c 8 \
  -n 08_llama70b_fs_noctx_leg \
  -o Decoder-Only/Llama-3.1-70B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --backend llama \
    --sheet LLAMA70B_FS_noCTX_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Llama-3.1-70B/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[225] Launching: LLAMA70B_FS_CTX1t_noLeg"
nlprun -g 4 -q sphinx -p standard -r 300G -c 8 \
  -n 09_llama70b_fs_ctx1t_noleg \
  -o Decoder-Only/Llama-3.1-70B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --backend llama \
    --sheet LLAMA70B_FS_CTX1t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Llama-3.1-70B/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[226] Launching: LLAMA70B_FS_CTX1t_Leg"
nlprun -g 4 -q sphinx -p standard -r 300G -c 8 \
  -n 10_llama70b_fs_ctx1t_leg \
  -o Decoder-Only/Llama-3.1-70B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --backend llama \
    --sheet LLAMA70B_FS_CTX1t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Llama-3.1-70B/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[227] Launching: LLAMA70B_FS_CTX2t_noLeg"
nlprun -g 4 -q sphinx -p standard -r 300G -c 8 \
  -n 11_llama70b_fs_ctx2t_noleg \
  -o Decoder-Only/Llama-3.1-70B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --backend llama \
    --sheet LLAMA70B_FS_CTX2t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Llama-3.1-70B/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[228] Launching: LLAMA70B_FS_CTX2t_Leg"
nlprun -g 4 -q sphinx -p standard -r 300G -c 8 \
  -n 12_llama70b_fs_ctx2t_leg \
  -o Decoder-Only/Llama-3.1-70B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --backend llama \
    --sheet LLAMA70B_FS_CTX2t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Llama-3.1-70B/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[229] Launching: LLAMA70B_ZScot_noCTX_noLeg"
nlprun -g 4 -q sphinx -p standard -r 300G -c 8 \
  -n 13_llama70b_zscot_noctx_noleg \
  -o Decoder-Only/Llama-3.1-70B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --backend llama \
    --sheet LLAMA70B_ZScot_noCTX_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Llama-3.1-70B/data \
    --dump_prompt"

echo "[230] Launching: LLAMA70B_ZScot_noCTX_Leg"
nlprun -g 4 -q sphinx -p standard -r 300G -c 8 \
  -n 14_llama70b_zscot_noctx_leg \
  -o Decoder-Only/Llama-3.1-70B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --backend llama \
    --sheet LLAMA70B_ZScot_noCTX_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Llama-3.1-70B/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[231] Launching: LLAMA70B_ZScot_CTX1t_noLeg"
nlprun -g 4 -q sphinx -p standard -r 300G -c 8 \
  -n 15_llama70b_zscot_ctx1t_noleg \
  -o Decoder-Only/Llama-3.1-70B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --backend llama \
    --sheet LLAMA70B_ZScot_CTX1t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Llama-3.1-70B/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[232] Launching: LLAMA70B_ZScot_CTX1t_Leg"
nlprun -g 4 -q sphinx -p standard -r 300G -c 8 \
  -n 16_llama70b_zscot_ctx1t_leg \
  -o Decoder-Only/Llama-3.1-70B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --backend llama \
    --sheet LLAMA70B_ZScot_CTX1t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Llama-3.1-70B/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[233] Launching: LLAMA70B_ZScot_CTX2t_noLeg"
nlprun -g 4 -q sphinx -p standard -r 300G -c 8 \
  -n 17_llama70b_zscot_ctx2t_noleg \
  -o Decoder-Only/Llama-3.1-70B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --backend llama \
    --sheet LLAMA70B_ZScot_CTX2t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Llama-3.1-70B/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[234] Launching: LLAMA70B_ZScot_CTX2t_Leg"
nlprun -g 4 -q sphinx -p standard -r 300G -c 8 \
  -n 18_llama70b_zscot_ctx2t_leg \
  -o Decoder-Only/Llama-3.1-70B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --backend llama \
    --sheet LLAMA70B_ZScot_CTX2t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Llama-3.1-70B/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[235] Launching: LLAMA70B_FScot_noCTX_noLeg"
nlprun -g 4 -q sphinx -p standard -r 300G -c 8 \
  -n 19_llama70b_fscot_noctx_noleg \
  -o Decoder-Only/Llama-3.1-70B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --backend llama \
    --sheet LLAMA70B_FScot_noCTX_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Llama-3.1-70B/data \
    --dump_prompt"

echo "[236] Launching: LLAMA70B_FScot_noCTX_Leg"
nlprun -g 4 -q sphinx -p standard -r 300G -c 8 \
  -n 20_llama70b_fscot_noctx_leg \
  -o Decoder-Only/Llama-3.1-70B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --backend llama \
    --sheet LLAMA70B_FScot_noCTX_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Llama-3.1-70B/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[237] Launching: LLAMA70B_FScot_CTX1t_noLeg"
nlprun -g 4 -q sphinx -p standard -r 300G -c 8 \
  -n 21_llama70b_fscot_ctx1t_noleg \
  -o Decoder-Only/Llama-3.1-70B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --backend llama \
    --sheet LLAMA70B_FScot_CTX1t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Llama-3.1-70B/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[238] Launching: LLAMA70B_FScot_CTX1t_Leg"
nlprun -g 4 -q sphinx -p standard -r 300G -c 8 \
  -n 22_llama70b_fscot_ctx1t_leg \
  -o Decoder-Only/Llama-3.1-70B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --backend llama \
    --sheet LLAMA70B_FScot_CTX1t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Llama-3.1-70B/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[239] Launching: LLAMA70B_FScot_CTX2t_noLeg"
nlprun -g 4 -q sphinx -p standard -r 300G -c 8 \
  -n 23_llama70b_fscot_ctx2t_noleg \
  -o Decoder-Only/Llama-3.1-70B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --backend llama \
    --sheet LLAMA70B_FScot_CTX2t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Llama-3.1-70B/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[240] Launching: LLAMA70B_FScot_CTX2t_Leg"
nlprun -g 4 -q sphinx -p standard -r 300G -c 8 \
  -n 24_llama70b_fscot_ctx2t_leg \
  -o Decoder-Only/Llama-3.1-70B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --backend llama \
    --sheet LLAMA70B_FScot_CTX2t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Llama-3.1-70B/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

fi

# ============================================================
# QWEN3_32B — Qwen/Qwen3-32B
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "qwen3_32b" ]]; then

mkdir -p Decoder-Only/Qwen3-32B/slurm_logs
mkdir -p Decoder-Only/Qwen3-32B/data

echo "[241] Launching: QWEN3_32B_ZS_noCTX_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 01_qwen3_32b_zs_noctx_noleg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_ZS_noCTX_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt"

echo "[242] Launching: QWEN3_32B_ZS_noCTX_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 02_qwen3_32b_zs_noctx_leg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_ZS_noCTX_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[243] Launching: QWEN3_32B_ZS_CTX1t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 03_qwen3_32b_zs_ctx1t_noleg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_ZS_CTX1t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[244] Launching: QWEN3_32B_ZS_CTX1t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 04_qwen3_32b_zs_ctx1t_leg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_ZS_CTX1t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[245] Launching: QWEN3_32B_ZS_CTX2t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 05_qwen3_32b_zs_ctx2t_noleg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_ZS_CTX2t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[246] Launching: QWEN3_32B_ZS_CTX2t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 06_qwen3_32b_zs_ctx2t_leg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_ZS_CTX2t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[247] Launching: QWEN3_32B_FS_noCTX_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 07_qwen3_32b_fs_noctx_noleg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_FS_noCTX_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt"

echo "[248] Launching: QWEN3_32B_FS_noCTX_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 08_qwen3_32b_fs_noctx_leg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_FS_noCTX_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[249] Launching: QWEN3_32B_FS_CTX1t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 09_qwen3_32b_fs_ctx1t_noleg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_FS_CTX1t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[250] Launching: QWEN3_32B_FS_CTX1t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 10_qwen3_32b_fs_ctx1t_leg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_FS_CTX1t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[251] Launching: QWEN3_32B_FS_CTX2t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 11_qwen3_32b_fs_ctx2t_noleg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_FS_CTX2t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[252] Launching: QWEN3_32B_FS_CTX2t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 12_qwen3_32b_fs_ctx2t_leg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_FS_CTX2t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[253] Launching: QWEN3_32B_ZScot_noCTX_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 13_qwen3_32b_zscot_noctx_noleg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_ZScot_noCTX_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt"

echo "[254] Launching: QWEN3_32B_ZScot_noCTX_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 14_qwen3_32b_zscot_noctx_leg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_ZScot_noCTX_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[255] Launching: QWEN3_32B_ZScot_CTX1t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 15_qwen3_32b_zscot_ctx1t_noleg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_ZScot_CTX1t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[256] Launching: QWEN3_32B_ZScot_CTX1t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 16_qwen3_32b_zscot_ctx1t_leg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_ZScot_CTX1t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[257] Launching: QWEN3_32B_ZScot_CTX2t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 17_qwen3_32b_zscot_ctx2t_noleg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_ZScot_CTX2t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[258] Launching: QWEN3_32B_ZScot_CTX2t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 18_qwen3_32b_zscot_ctx2t_leg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_ZScot_CTX2t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[259] Launching: QWEN3_32B_FScot_noCTX_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 19_qwen3_32b_fscot_noctx_noleg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_FScot_noCTX_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt"

echo "[260] Launching: QWEN3_32B_FScot_noCTX_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 20_qwen3_32b_fscot_noctx_leg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_FScot_noCTX_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[261] Launching: QWEN3_32B_FScot_CTX1t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 21_qwen3_32b_fscot_ctx1t_noleg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_FScot_CTX1t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[262] Launching: QWEN3_32B_FScot_CTX1t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 22_qwen3_32b_fscot_ctx1t_leg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_FScot_CTX1t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[263] Launching: QWEN3_32B_FScot_CTX2t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 23_qwen3_32b_fscot_ctx2t_noleg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_FScot_CTX2t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[264] Launching: QWEN3_32B_FScot_CTX2t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 24_qwen3_32b_fscot_ctx2t_leg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_FScot_CTX2t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

fi

# ============================================================
# QWEN3_32B_THINK — Qwen/Qwen3-32B
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "qwen3_32b_think" ]]; then

mkdir -p Decoder-Only/Qwen3-32B-Thinking/slurm_logs
mkdir -p Decoder-Only/Qwen3-32B-Thinking/data

echo "[265] Launching: QWEN3_32B_THINK_ZS_noCTX_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 01_qwen3_32b_think_zs_noctx_noleg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32B_THINK_ZS_noCTX_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt"

echo "[266] Launching: QWEN3_32B_THINK_ZS_noCTX_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 02_qwen3_32b_think_zs_noctx_leg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32B_THINK_ZS_noCTX_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[267] Launching: QWEN3_32B_THINK_ZS_CTX1t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 03_qwen3_32b_think_zs_ctx1t_noleg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32B_THINK_ZS_CTX1t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[268] Launching: QWEN3_32B_THINK_ZS_CTX1t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 04_qwen3_32b_think_zs_ctx1t_leg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32B_THINK_ZS_CTX1t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[269] Launching: QWEN3_32B_THINK_ZS_CTX2t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 05_qwen3_32b_think_zs_ctx2t_noleg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32B_THINK_ZS_CTX2t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[270] Launching: QWEN3_32B_THINK_ZS_CTX2t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 06_qwen3_32b_think_zs_ctx2t_leg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32B_THINK_ZS_CTX2t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[271] Launching: QWEN3_32B_THINK_FS_noCTX_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 07_qwen3_32b_think_fs_noctx_noleg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32B_THINK_FS_noCTX_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt"

echo "[272] Launching: QWEN3_32B_THINK_FS_noCTX_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 08_qwen3_32b_think_fs_noctx_leg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32B_THINK_FS_noCTX_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[273] Launching: QWEN3_32B_THINK_FS_CTX1t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 09_qwen3_32b_think_fs_ctx1t_noleg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32B_THINK_FS_CTX1t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[274] Launching: QWEN3_32B_THINK_FS_CTX1t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 10_qwen3_32b_think_fs_ctx1t_leg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32B_THINK_FS_CTX1t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[275] Launching: QWEN3_32B_THINK_FS_CTX2t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 11_qwen3_32b_think_fs_ctx2t_noleg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32B_THINK_FS_CTX2t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[276] Launching: QWEN3_32B_THINK_FS_CTX2t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 12_qwen3_32b_think_fs_ctx2t_leg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32B_THINK_FS_CTX2t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[277] Launching: QWEN3_32B_THINK_ZScot_noCTX_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 13_qwen3_32b_think_zscot_noctx_noleg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32B_THINK_ZScot_noCTX_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt"

echo "[278] Launching: QWEN3_32B_THINK_ZScot_noCTX_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 14_qwen3_32b_think_zscot_noctx_leg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32B_THINK_ZScot_noCTX_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[279] Launching: QWEN3_32B_THINK_ZScot_CTX1t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 15_qwen3_32b_think_zscot_ctx1t_noleg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32B_THINK_ZScot_CTX1t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[280] Launching: QWEN3_32B_THINK_ZScot_CTX1t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 16_qwen3_32b_think_zscot_ctx1t_leg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32B_THINK_ZScot_CTX1t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[281] Launching: QWEN3_32B_THINK_ZScot_CTX2t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 17_qwen3_32b_think_zscot_ctx2t_noleg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32B_THINK_ZScot_CTX2t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[282] Launching: QWEN3_32B_THINK_ZScot_CTX2t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 18_qwen3_32b_think_zscot_ctx2t_leg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32B_THINK_ZScot_CTX2t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[283] Launching: QWEN3_32B_THINK_FScot_noCTX_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 19_qwen3_32b_think_fscot_noctx_noleg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32B_THINK_FScot_noCTX_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt"

echo "[284] Launching: QWEN3_32B_THINK_FScot_noCTX_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 20_qwen3_32b_think_fscot_noctx_leg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32B_THINK_FScot_noCTX_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[285] Launching: QWEN3_32B_THINK_FScot_CTX1t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 21_qwen3_32b_think_fscot_ctx1t_noleg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32B_THINK_FScot_CTX1t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[286] Launching: QWEN3_32B_THINK_FScot_CTX1t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 22_qwen3_32b_think_fscot_ctx1t_leg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32B_THINK_FScot_CTX1t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[287] Launching: QWEN3_32B_THINK_FScot_CTX2t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 23_qwen3_32b_think_fscot_ctx2t_noleg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32B_THINK_FScot_CTX2t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[288] Launching: QWEN3_32B_THINK_FScot_CTX2t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 24_qwen3_32b_think_fscot_ctx2t_leg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32B_THINK_FScot_CTX2t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

fi

# ============================================================
# QWQ_32B — Qwen/QwQ-32B
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "qwq_32b" ]]; then

mkdir -p Decoder-Only/QwQ-32B/slurm_logs
mkdir -p Decoder-Only/QwQ-32B/data

echo "[289] Launching: QWQ_32B_ZS_noCTX_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 01_qwq_32b_zs_noctx_noleg \
  -o Decoder-Only/QwQ-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/QwQ-32B \
    --backend qwq \
    --sheet QWQ_32B_ZS_noCTX_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/QwQ-32B/data \
    --dump_prompt"

echo "[290] Launching: QWQ_32B_ZS_noCTX_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 02_qwq_32b_zs_noctx_leg \
  -o Decoder-Only/QwQ-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/QwQ-32B \
    --backend qwq \
    --sheet QWQ_32B_ZS_noCTX_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/QwQ-32B/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[291] Launching: QWQ_32B_ZS_CTX1t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 03_qwq_32b_zs_ctx1t_noleg \
  -o Decoder-Only/QwQ-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/QwQ-32B \
    --backend qwq \
    --sheet QWQ_32B_ZS_CTX1t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/QwQ-32B/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[292] Launching: QWQ_32B_ZS_CTX1t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 04_qwq_32b_zs_ctx1t_leg \
  -o Decoder-Only/QwQ-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/QwQ-32B \
    --backend qwq \
    --sheet QWQ_32B_ZS_CTX1t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/QwQ-32B/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[293] Launching: QWQ_32B_ZS_CTX2t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 05_qwq_32b_zs_ctx2t_noleg \
  -o Decoder-Only/QwQ-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/QwQ-32B \
    --backend qwq \
    --sheet QWQ_32B_ZS_CTX2t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/QwQ-32B/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[294] Launching: QWQ_32B_ZS_CTX2t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 06_qwq_32b_zs_ctx2t_leg \
  -o Decoder-Only/QwQ-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/QwQ-32B \
    --backend qwq \
    --sheet QWQ_32B_ZS_CTX2t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/QwQ-32B/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[295] Launching: QWQ_32B_FS_noCTX_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 07_qwq_32b_fs_noctx_noleg \
  -o Decoder-Only/QwQ-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/QwQ-32B \
    --backend qwq \
    --sheet QWQ_32B_FS_noCTX_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/QwQ-32B/data \
    --dump_prompt"

echo "[296] Launching: QWQ_32B_FS_noCTX_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 08_qwq_32b_fs_noctx_leg \
  -o Decoder-Only/QwQ-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/QwQ-32B \
    --backend qwq \
    --sheet QWQ_32B_FS_noCTX_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/QwQ-32B/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[297] Launching: QWQ_32B_FS_CTX1t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 09_qwq_32b_fs_ctx1t_noleg \
  -o Decoder-Only/QwQ-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/QwQ-32B \
    --backend qwq \
    --sheet QWQ_32B_FS_CTX1t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/QwQ-32B/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[298] Launching: QWQ_32B_FS_CTX1t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 10_qwq_32b_fs_ctx1t_leg \
  -o Decoder-Only/QwQ-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/QwQ-32B \
    --backend qwq \
    --sheet QWQ_32B_FS_CTX1t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/QwQ-32B/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[299] Launching: QWQ_32B_FS_CTX2t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 11_qwq_32b_fs_ctx2t_noleg \
  -o Decoder-Only/QwQ-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/QwQ-32B \
    --backend qwq \
    --sheet QWQ_32B_FS_CTX2t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/QwQ-32B/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[300] Launching: QWQ_32B_FS_CTX2t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 12_qwq_32b_fs_ctx2t_leg \
  -o Decoder-Only/QwQ-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/QwQ-32B \
    --backend qwq \
    --sheet QWQ_32B_FS_CTX2t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/QwQ-32B/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[301] Launching: QWQ_32B_ZScot_noCTX_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 13_qwq_32b_zscot_noctx_noleg \
  -o Decoder-Only/QwQ-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/QwQ-32B \
    --backend qwq \
    --sheet QWQ_32B_ZScot_noCTX_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/QwQ-32B/data \
    --dump_prompt"

echo "[302] Launching: QWQ_32B_ZScot_noCTX_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 14_qwq_32b_zscot_noctx_leg \
  -o Decoder-Only/QwQ-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/QwQ-32B \
    --backend qwq \
    --sheet QWQ_32B_ZScot_noCTX_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/QwQ-32B/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[303] Launching: QWQ_32B_ZScot_CTX1t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 15_qwq_32b_zscot_ctx1t_noleg \
  -o Decoder-Only/QwQ-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/QwQ-32B \
    --backend qwq \
    --sheet QWQ_32B_ZScot_CTX1t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/QwQ-32B/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[304] Launching: QWQ_32B_ZScot_CTX1t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 16_qwq_32b_zscot_ctx1t_leg \
  -o Decoder-Only/QwQ-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/QwQ-32B \
    --backend qwq \
    --sheet QWQ_32B_ZScot_CTX1t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/QwQ-32B/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[305] Launching: QWQ_32B_ZScot_CTX2t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 17_qwq_32b_zscot_ctx2t_noleg \
  -o Decoder-Only/QwQ-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/QwQ-32B \
    --backend qwq \
    --sheet QWQ_32B_ZScot_CTX2t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/QwQ-32B/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[306] Launching: QWQ_32B_ZScot_CTX2t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 18_qwq_32b_zscot_ctx2t_leg \
  -o Decoder-Only/QwQ-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/QwQ-32B \
    --backend qwq \
    --sheet QWQ_32B_ZScot_CTX2t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/QwQ-32B/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[307] Launching: QWQ_32B_FScot_noCTX_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 19_qwq_32b_fscot_noctx_noleg \
  -o Decoder-Only/QwQ-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/QwQ-32B \
    --backend qwq \
    --sheet QWQ_32B_FScot_noCTX_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/QwQ-32B/data \
    --dump_prompt"

echo "[308] Launching: QWQ_32B_FScot_noCTX_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 20_qwq_32b_fscot_noctx_leg \
  -o Decoder-Only/QwQ-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/QwQ-32B \
    --backend qwq \
    --sheet QWQ_32B_FScot_noCTX_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/QwQ-32B/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[309] Launching: QWQ_32B_FScot_CTX1t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 21_qwq_32b_fscot_ctx1t_noleg \
  -o Decoder-Only/QwQ-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/QwQ-32B \
    --backend qwq \
    --sheet QWQ_32B_FScot_CTX1t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/QwQ-32B/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[310] Launching: QWQ_32B_FScot_CTX1t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 22_qwq_32b_fscot_ctx1t_leg \
  -o Decoder-Only/QwQ-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/QwQ-32B \
    --backend qwq \
    --sheet QWQ_32B_FScot_CTX1t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/QwQ-32B/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[311] Launching: QWQ_32B_FScot_CTX2t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 23_qwq_32b_fscot_ctx2t_noleg \
  -o Decoder-Only/QwQ-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/QwQ-32B \
    --backend qwq \
    --sheet QWQ_32B_FScot_CTX2t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/QwQ-32B/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[312] Launching: QWQ_32B_FScot_CTX2t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n 24_qwq_32b_fscot_ctx2t_leg \
  -o Decoder-Only/QwQ-32B/slurm_logs/%x.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --gold Datasets/FullTest_Final.xlsx \
    --model Qwen/QwQ-32B \
    --backend qwq \
    --sheet QWQ_32B_FScot_CTX2t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/QwQ-32B/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

fi

echo "Done. Submitted jobs for $MODEL_FILTER."
