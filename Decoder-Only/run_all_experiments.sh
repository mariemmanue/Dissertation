#!/bin/bash
# Auto-generated experimental run script
# Total jobs: 72
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
#   ./run_all_experiments.sh           # launch all 72 jobs
#   ./run_all_experiments.sh phi4      # launch only phi4 jobs (24)
#   ./run_all_experiments.sh gemini    # launch only gemini jobs (24)
#   ./run_all_experiments.sh gpt4o     # launch only gpt4o jobs (24)

set -e

MODEL_FILTER="${1:-all}"

# ============================================================
# PHI4 — microsoft/phi-4
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "phi4" ]]; then

echo "[01] Launching: PHI4_ZS_noCTX_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n phi4_zs_noctx_noleg \
  -o Decoder-Only/Phi-4/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_ZS_noCTX_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt"

echo "[02] Launching: PHI4_ZS_noCTX_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n phi4_zs_noctx_leg \
  -o Decoder-Only/Phi-4/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_ZS_noCTX_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[03] Launching: PHI4_ZS_CTX1t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n phi4_zs_ctx1t_noleg \
  -o Decoder-Only/Phi-4/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[04] Launching: PHI4_ZS_CTX1t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n phi4_zs_ctx1t_leg \
  -o Decoder-Only/Phi-4/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[05] Launching: PHI4_ZS_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n phi4_zs_ctx2t_noleg \
  -o Decoder-Only/Phi-4/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[06] Launching: PHI4_ZS_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n phi4_zs_ctx2t_leg \
  -o Decoder-Only/Phi-4/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[07] Launching: PHI4_FS_noCTX_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n phi4_fs_noctx_noleg \
  -o Decoder-Only/Phi-4/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_FS_noCTX_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt"

echo "[08] Launching: PHI4_FS_noCTX_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n phi4_fs_noctx_leg \
  -o Decoder-Only/Phi-4/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_FS_noCTX_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[09] Launching: PHI4_FS_CTX1t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n phi4_fs_ctx1t_noleg \
  -o Decoder-Only/Phi-4/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[10] Launching: PHI4_FS_CTX1t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n phi4_fs_ctx1t_leg \
  -o Decoder-Only/Phi-4/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[11] Launching: PHI4_FS_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n phi4_fs_ctx2t_noleg \
  -o Decoder-Only/Phi-4/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[12] Launching: PHI4_FS_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n phi4_fs_ctx2t_leg \
  -o Decoder-Only/Phi-4/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[13] Launching: PHI4_ZScot_noCTX_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n phi4_zscot_noctx_noleg \
  -o Decoder-Only/Phi-4/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_ZScot_noCTX_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt"

echo "[14] Launching: PHI4_ZScot_noCTX_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n phi4_zscot_noctx_leg \
  -o Decoder-Only/Phi-4/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_ZScot_noCTX_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[15] Launching: PHI4_ZScot_CTX1t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n phi4_zscot_ctx1t_noleg \
  -o Decoder-Only/Phi-4/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[16] Launching: PHI4_ZScot_CTX1t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n phi4_zscot_ctx1t_leg \
  -o Decoder-Only/Phi-4/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[17] Launching: PHI4_ZScot_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n phi4_zscot_ctx2t_noleg \
  -o Decoder-Only/Phi-4/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[18] Launching: PHI4_ZScot_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n phi4_zscot_ctx2t_leg \
  -o Decoder-Only/Phi-4/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[19] Launching: PHI4_FScot_noCTX_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n phi4_fscot_noctx_noleg \
  -o Decoder-Only/Phi-4/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_FScot_noCTX_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt"

echo "[20] Launching: PHI4_FScot_noCTX_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n phi4_fscot_noctx_leg \
  -o Decoder-Only/Phi-4/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_FScot_noCTX_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[21] Launching: PHI4_FScot_CTX1t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n phi4_fscot_ctx1t_noleg \
  -o Decoder-Only/Phi-4/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[22] Launching: PHI4_FScot_CTX1t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n phi4_fscot_ctx1t_leg \
  -o Decoder-Only/Phi-4/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[23] Launching: PHI4_FScot_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n phi4_fscot_ctx2t_noleg \
  -o Decoder-Only/Phi-4/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[24] Launching: PHI4_FScot_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n phi4_fscot_ctx2t_leg \
  -o Decoder-Only/Phi-4/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[25] Launching: GEMINI_ZS_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gemini_zs_noctx_noleg \
  -o Decoder-Only/Gemini/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_ZS_noCTX_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt"

echo "[26] Launching: GEMINI_ZS_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gemini_zs_noctx_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_ZS_noCTX_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[27] Launching: GEMINI_ZS_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gemini_zs_ctx1t_noleg \
  -o Decoder-Only/Gemini/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[28] Launching: GEMINI_ZS_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gemini_zs_ctx1t_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[29] Launching: GEMINI_ZS_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gemini_zs_ctx2t_noleg \
  -o Decoder-Only/Gemini/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[30] Launching: GEMINI_ZS_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gemini_zs_ctx2t_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[31] Launching: GEMINI_FS_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gemini_fs_noctx_noleg \
  -o Decoder-Only/Gemini/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_FS_noCTX_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt"

echo "[32] Launching: GEMINI_FS_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gemini_fs_noctx_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_FS_noCTX_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[33] Launching: GEMINI_FS_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gemini_fs_ctx1t_noleg \
  -o Decoder-Only/Gemini/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[34] Launching: GEMINI_FS_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gemini_fs_ctx1t_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[35] Launching: GEMINI_FS_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gemini_fs_ctx2t_noleg \
  -o Decoder-Only/Gemini/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[36] Launching: GEMINI_FS_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gemini_fs_ctx2t_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[37] Launching: GEMINI_ZScot_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gemini_zscot_noctx_noleg \
  -o Decoder-Only/Gemini/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_ZScot_noCTX_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt"

echo "[38] Launching: GEMINI_ZScot_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gemini_zscot_noctx_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_ZScot_noCTX_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[39] Launching: GEMINI_ZScot_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gemini_zscot_ctx1t_noleg \
  -o Decoder-Only/Gemini/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[40] Launching: GEMINI_ZScot_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gemini_zscot_ctx1t_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[41] Launching: GEMINI_ZScot_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gemini_zscot_ctx2t_noleg \
  -o Decoder-Only/Gemini/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[42] Launching: GEMINI_ZScot_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gemini_zscot_ctx2t_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[43] Launching: GEMINI_FScot_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gemini_fscot_noctx_noleg \
  -o Decoder-Only/Gemini/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_FScot_noCTX_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt"

echo "[44] Launching: GEMINI_FScot_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gemini_fscot_noctx_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_FScot_noCTX_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[45] Launching: GEMINI_FScot_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gemini_fscot_ctx1t_noleg \
  -o Decoder-Only/Gemini/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[46] Launching: GEMINI_FScot_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gemini_fscot_ctx1t_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[47] Launching: GEMINI_FScot_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gemini_fscot_ctx2t_noleg \
  -o Decoder-Only/Gemini/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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

echo "[48] Launching: GEMINI_FScot_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gemini_fscot_ctx2t_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
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
# GPT4O — gpt-4o
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "gpt4o" ]]; then

echo "[49] Launching: GPT4O_ZS_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gpt4o_zs_noctx_noleg \
  -o Decoder-Only/GPT4o/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gpt-4o \
    --backend openai \
    --sheet GPT4O_ZS_noCTX_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT4o/data \
    --dump_prompt"

echo "[50] Launching: GPT4O_ZS_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gpt4o_zs_noctx_leg \
  -o Decoder-Only/GPT4o/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gpt-4o \
    --backend openai \
    --sheet GPT4O_ZS_noCTX_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT4o/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[51] Launching: GPT4O_ZS_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gpt4o_zs_ctx1t_noleg \
  -o Decoder-Only/GPT4o/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gpt-4o \
    --backend openai \
    --sheet GPT4O_ZS_CTX1t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT4o/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[52] Launching: GPT4O_ZS_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gpt4o_zs_ctx1t_leg \
  -o Decoder-Only/GPT4o/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gpt-4o \
    --backend openai \
    --sheet GPT4O_ZS_CTX1t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT4o/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[53] Launching: GPT4O_ZS_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gpt4o_zs_ctx2t_noleg \
  -o Decoder-Only/GPT4o/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gpt-4o \
    --backend openai \
    --sheet GPT4O_ZS_CTX2t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT4o/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[54] Launching: GPT4O_ZS_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gpt4o_zs_ctx2t_leg \
  -o Decoder-Only/GPT4o/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gpt-4o \
    --backend openai \
    --sheet GPT4O_ZS_CTX2t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT4o/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[55] Launching: GPT4O_FS_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gpt4o_fs_noctx_noleg \
  -o Decoder-Only/GPT4o/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gpt-4o \
    --backend openai \
    --sheet GPT4O_FS_noCTX_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT4o/data \
    --dump_prompt"

echo "[56] Launching: GPT4O_FS_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gpt4o_fs_noctx_leg \
  -o Decoder-Only/GPT4o/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gpt-4o \
    --backend openai \
    --sheet GPT4O_FS_noCTX_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT4o/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[57] Launching: GPT4O_FS_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gpt4o_fs_ctx1t_noleg \
  -o Decoder-Only/GPT4o/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gpt-4o \
    --backend openai \
    --sheet GPT4O_FS_CTX1t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT4o/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[58] Launching: GPT4O_FS_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gpt4o_fs_ctx1t_leg \
  -o Decoder-Only/GPT4o/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gpt-4o \
    --backend openai \
    --sheet GPT4O_FS_CTX1t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT4o/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[59] Launching: GPT4O_FS_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gpt4o_fs_ctx2t_noleg \
  -o Decoder-Only/GPT4o/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gpt-4o \
    --backend openai \
    --sheet GPT4O_FS_CTX2t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT4o/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[60] Launching: GPT4O_FS_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gpt4o_fs_ctx2t_leg \
  -o Decoder-Only/GPT4o/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gpt-4o \
    --backend openai \
    --sheet GPT4O_FS_CTX2t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT4o/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[61] Launching: GPT4O_ZScot_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gpt4o_zscot_noctx_noleg \
  -o Decoder-Only/GPT4o/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gpt-4o \
    --backend openai \
    --sheet GPT4O_ZScot_noCTX_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT4o/data \
    --dump_prompt"

echo "[62] Launching: GPT4O_ZScot_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gpt4o_zscot_noctx_leg \
  -o Decoder-Only/GPT4o/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gpt-4o \
    --backend openai \
    --sheet GPT4O_ZScot_noCTX_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT4o/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[63] Launching: GPT4O_ZScot_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gpt4o_zscot_ctx1t_noleg \
  -o Decoder-Only/GPT4o/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gpt-4o \
    --backend openai \
    --sheet GPT4O_ZScot_CTX1t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT4o/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[64] Launching: GPT4O_ZScot_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gpt4o_zscot_ctx1t_leg \
  -o Decoder-Only/GPT4o/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gpt-4o \
    --backend openai \
    --sheet GPT4O_ZScot_CTX1t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT4o/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[65] Launching: GPT4O_ZScot_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gpt4o_zscot_ctx2t_noleg \
  -o Decoder-Only/GPT4o/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gpt-4o \
    --backend openai \
    --sheet GPT4O_ZScot_CTX2t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT4o/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[66] Launching: GPT4O_ZScot_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gpt4o_zscot_ctx2t_leg \
  -o Decoder-Only/GPT4o/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gpt-4o \
    --backend openai \
    --sheet GPT4O_ZScot_CTX2t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT4o/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

echo "[67] Launching: GPT4O_FScot_noCTX_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gpt4o_fscot_noctx_noleg \
  -o Decoder-Only/GPT4o/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gpt-4o \
    --backend openai \
    --sheet GPT4O_FScot_noCTX_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT4o/data \
    --dump_prompt"

echo "[68] Launching: GPT4O_FScot_noCTX_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gpt4o_fscot_noctx_leg \
  -o Decoder-Only/GPT4o/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gpt-4o \
    --backend openai \
    --sheet GPT4O_FScot_noCTX_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT4o/data \
    --dump_prompt \
    --dialect_legitimacy"

echo "[69] Launching: GPT4O_FScot_CTX1t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gpt4o_fscot_ctx1t_noleg \
  -o Decoder-Only/GPT4o/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gpt-4o \
    --backend openai \
    --sheet GPT4O_FScot_CTX1t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT4o/data \
    --dump_prompt \
    --context \
    --context_mode single_turn"

echo "[70] Launching: GPT4O_FScot_CTX1t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gpt4o_fscot_ctx1t_leg \
  -o Decoder-Only/GPT4o/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gpt-4o \
    --backend openai \
    --sheet GPT4O_FScot_CTX1t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT4o/data \
    --dump_prompt \
    --context \
    --context_mode single_turn \
    --dialect_legitimacy"

echo "[71] Launching: GPT4O_FScot_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gpt4o_fscot_ctx2t_noleg \
  -o Decoder-Only/GPT4o/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gpt-4o \
    --backend openai \
    --sheet GPT4O_FScot_CTX2t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT4o/data \
    --dump_prompt \
    --context \
    --context_mode two_turn"

echo "[72] Launching: GPT4O_FScot_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n gpt4o_fscot_ctx2t_leg \
  -o Decoder-Only/GPT4o/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
    --model gpt-4o \
    --backend openai \
    --sheet GPT4O_FScot_CTX2t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format markdown \
    --output_dir Decoder-Only/GPT4o/data \
    --dump_prompt \
    --context \
    --context_mode two_turn \
    --dialect_legitimacy"

fi

echo "Done. Submitted jobs for $MODEL_FILTER."
