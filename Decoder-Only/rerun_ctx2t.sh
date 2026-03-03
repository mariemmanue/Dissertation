#!/bin/bash
# Re-run all CTX2t (wide context) experiments after the assistant-ack fix.
#
# Non-reasoning models:  phi4, qwen25_7b, qwen3_32b, gpt41, gpt52, gemini, gemini25_pro, g3flash_nothink
# Reasoning models:      qwen3_32bthinking, gemini3_pro, g3flash, gemini25_pro_think,
#                        gpt5, o4mini, o3, o3mini, o3deep, o4miniDeep
# Grid:   4 instruction_types × 1 context(wide) × 2 legitimacy = 8 per model
# Total:  136 jobs
#
# Usage:
#   chmod +x Decoder-Only/rerun_ctx2t.sh
#   ./Decoder-Only/rerun_ctx2t.sh                         # all 136 jobs on jag (default)
#   ./Decoder-Only/rerun_ctx2t.sh all john                # all API-model jobs on john (CPU nodes)
#   ./Decoder-Only/rerun_ctxwide.sh phi4                  # launch only phi4 CTX2t jobs (8)
#   ./Decoder-Only/rerun_ctxwide.sh gemini                # launch only Gemini 2.5 Flash jobs (8)
#   ./Decoder-Only/rerun_ctxwide.sh qwen3_32bthinking     # launch only Qwen3-32B thinking jobs (8)
#   ./Decoder-Only/rerun_ctxwide.sh o4mini                # launch only o4-mini reasoning jobs (8)
#   ./Decoder-Only/rerun_ctxwide.sh o3                    # launch only o3 reasoning jobs (8)
#   ./Decoder-Only/rerun_ctxwide.sh o3mini                # launch only o3-mini reasoning jobs (8)
#   ./Decoder-Only/rerun_ctxwide.sh gpt52                 # launch only GPT-5.2 non-reasoning jobs (8)
#   ./Decoder-Only/rerun_ctxwide.sh gemini3_flash_nothink # launch only Gemini 3 Flash no-thinking jobs (8)
#   ./Decoder-Only/rerun_ctxwide.sh gemini3_flash         # launch only Gemini 3 Flash thinking jobs (8)
#   ./Decoder-Only/rerun_ctxwide.sh gemini25_pro          # launch only Gemini 2.5 Pro reasoning jobs (8)
#   ./Decoder-Only/rerun_ctxwide.sh gpt5                  # launch only GPT-5 reasoning jobs (8)
#   ./Decoder-Only/rerun_ctxwide.sh o3deep                # launch only o3-deep-research jobs (8)
#   ./Decoder-Only/rerun_ctxwide.sh o4miniDeep            # launch only o4-mini-deep-research jobs (8)

set -e

MODEL_FILTER="${1:-all}"
API_QUEUE="${2:-jag}"     # override: pass "john" for CPU-only nodes (API models only)

BASE_DIR="/nlp/scr/mtano/Dissertation"
CONDA_INIT=". /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="cgedit"
INPUT_FILE="Datasets/FullTest_Final.xlsx"
GOLD_FILE="Datasets/FullTest_Final.xlsx"
OUTPUT_FORMAT="markdown"

JOB=0

# ============================================================
# PHI4 — microsoft/phi-4
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "phi4" ]]; then

mkdir -p Decoder-Only/Phi-4/slurm_logs
mkdir -p Decoder-Only/Phi-4/data

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: PHI4_ZS_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_phi4_zs_ctxwide_noleg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_ZS_CTX2t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: PHI4_ZS_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_phi4_zs_ctxwide_leg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_ZS_CTX2t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: PHI4_FS_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_phi4_fs_ctxwide_noleg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_FS_CTX2t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: PHI4_FS_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_phi4_fs_ctxwide_leg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_FS_CTX2t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: PHI4_ZScot_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_phi4_zscot_ctxwide_noleg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_ZScot_CTX2t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: PHI4_ZScot_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_phi4_zscot_ctxwide_leg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_ZScot_CTX2t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: PHI4_FScot_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_phi4_fscot_ctxwide_noleg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_FScot_CTX2t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: PHI4_FScot_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_phi4_fscot_ctxwide_leg \
  -o Decoder-Only/Phi-4/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model microsoft/phi-4 \
    --backend phi \
    --sheet PHI4_FScot_CTX2t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Phi-4/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

fi

# ============================================================
# GEMINI — gemini-2.5-flash
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "gemini" ]]; then

mkdir -p Decoder-Only/Gemini/slurm_logs
mkdir -p Decoder-Only/Gemini/data

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GEMINI_ZS_CTX2t_noLeg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_gemini_zs_ctxwide_noleg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_ZS_CTX2t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GEMINI_ZS_CTX2t_Leg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_gemini_zs_ctxwide_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_ZS_CTX2t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GEMINI_FS_CTX2t_noLeg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_gemini_fs_ctxwide_noleg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_FS_CTX2t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GEMINI_FS_CTX2t_Leg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_gemini_fs_ctxwide_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_FS_CTX2t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GEMINI_ZScot_CTX2t_noLeg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_gemini_zscot_ctxwide_noleg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_ZScot_CTX2t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GEMINI_ZScot_CTX2t_Leg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_gemini_zscot_ctxwide_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_ZScot_CTX2t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GEMINI_FScot_CTX2t_noLeg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_gemini_fscot_ctxwide_noleg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_FScot_CTX2t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GEMINI_FScot_CTX2t_Leg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_gemini_fscot_ctxwide_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model gemini-2.5-flash \
    --backend gemini \
    --sheet GEMINI_FScot_CTX2t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Gemini/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

fi

# ============================================================
# GEMINI3_PRO — gemini-3.1-pro-preview
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "gemini3_pro" ]]; then

mkdir -p Decoder-Only/Gemini3_Pro/slurm_logs
mkdir -p Decoder-Only/Gemini3_Pro/data

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GEMINI3_PRO_ZS_CTX2t_noLeg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_gemini3_pro_zs_ctxwide_noleg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model gemini-3.1-pro-preview \
    --backend gemini3 \
    --thinking_level high \
    --sheet GEMINI3_PRO_ZS_CTX2t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GEMINI3_PRO_ZS_CTX2t_Leg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_gemini3_pro_zs_ctxwide_leg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model gemini-3.1-pro-preview \
    --backend gemini3 \
    --thinking_level high \
    --sheet GEMINI3_PRO_ZS_CTX2t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GEMINI3_PRO_FS_CTX2t_noLeg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_gemini3_pro_fs_ctxwide_noleg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model gemini-3.1-pro-preview \
    --backend gemini3 \
    --thinking_level high \
    --sheet GEMINI3_PRO_FS_CTX2t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GEMINI3_PRO_FS_CTX2t_Leg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_gemini3_pro_fs_ctxwide_leg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model gemini-3.1-pro-preview \
    --backend gemini3 \
    --thinking_level high \
    --sheet GEMINI3_PRO_FS_CTX2t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GEMINI3_PRO_ZScot_CTX2t_noLeg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_gemini3_pro_zscot_ctxwide_noleg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model gemini-3.1-pro-preview \
    --backend gemini3 \
    --thinking_level high \
    --sheet GEMINI3_PRO_ZScot_CTX2t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GEMINI3_PRO_ZScot_CTX2t_Leg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_gemini3_pro_zscot_ctxwide_leg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model gemini-3.1-pro-preview \
    --backend gemini3 \
    --thinking_level high \
    --sheet GEMINI3_PRO_ZScot_CTX2t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GEMINI3_PRO_FScot_CTX2t_noLeg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_gemini3_pro_fscot_ctxwide_noleg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model gemini-3.1-pro-preview \
    --backend gemini3 \
    --thinking_level high \
    --sheet GEMINI3_PRO_FScot_CTX2t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GEMINI3_PRO_FScot_CTX2t_Leg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_gemini3_pro_fscot_ctxwide_leg \
  -o Decoder-Only/Gemini3_Pro/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model gemini-3.1-pro-preview \
    --backend gemini3 \
    --thinking_level high \
    --sheet GEMINI3_PRO_FScot_CTX2t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Gemini3_Pro/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

fi

# ============================================================
# QWEN25_7B — Qwen/Qwen2.5-7B-Instruct
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "qwen25_7b" ]]; then

mkdir -p Decoder-Only/Qwen2.5/slurm_logs
mkdir -p Decoder-Only/Qwen2.5/data

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN25_7B_ZS_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_qwen25_7b_zs_ctxwide_noleg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_ZS_CTX2t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN25_7B_ZS_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_qwen25_7b_zs_ctxwide_leg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_ZS_CTX2t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN25_7B_FS_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_qwen25_7b_fs_ctxwide_noleg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_FS_CTX2t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN25_7B_FS_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_qwen25_7b_fs_ctxwide_leg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_FS_CTX2t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN25_7B_ZScot_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_qwen25_7b_zscot_ctxwide_noleg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_ZScot_CTX2t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN25_7B_ZScot_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_qwen25_7b_zscot_ctxwide_leg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_ZScot_CTX2t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN25_7B_FScot_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_qwen25_7b_fscot_ctxwide_noleg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_FScot_CTX2t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN25_7B_FScot_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_qwen25_7b_fscot_ctxwide_leg \
  -o Decoder-Only/Qwen2.5/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend qwen \
    --sheet QWEN25_7B_FScot_CTX2t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Qwen2.5/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

fi

# ============================================================
# GPT41 — gpt-4.1
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "gpt41" ]]; then

mkdir -p Decoder-Only/GPT41/slurm_logs
mkdir -p Decoder-Only/GPT41/data

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GPT41_ZS_CTX2t_noLeg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_gpt41_zs_ctxwide_noleg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_ZS_CTX2t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GPT41_ZS_CTX2t_Leg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_gpt41_zs_ctxwide_leg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_ZS_CTX2t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GPT41_FS_CTX2t_noLeg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_gpt41_fs_ctxwide_noleg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_FS_CTX2t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GPT41_FS_CTX2t_Leg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_gpt41_fs_ctxwide_leg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_FS_CTX2t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GPT41_ZScot_CTX2t_noLeg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_gpt41_zscot_ctxwide_noleg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_ZScot_CTX2t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GPT41_ZScot_CTX2t_Leg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_gpt41_zscot_ctxwide_leg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_ZScot_CTX2t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GPT41_FScot_CTX2t_noLeg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_gpt41_fscot_ctxwide_noleg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_FScot_CTX2t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GPT41_FScot_CTX2t_Leg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_gpt41_fscot_ctxwide_leg \
  -o Decoder-Only/GPT41/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model gpt-4.1 \
    --backend openai \
    --sheet GPT41_FScot_CTX2t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/GPT41/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

fi

# ============================================================
# QWEN3_32B — Qwen/Qwen3-32B
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "qwen3_32b" ]]; then

mkdir -p Decoder-Only/Qwen3-32B/slurm_logs
mkdir -p Decoder-Only/Qwen3-32B/data

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN3_32B_ZS_CTX2t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n ${JOB}_qwen3_32b_zs_ctxwide_noleg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_ZS_CTX2t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN3_32B_ZS_CTX2t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n ${JOB}_qwen3_32b_zs_ctxwide_leg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_ZS_CTX2t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN3_32B_FS_CTX2t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n ${JOB}_qwen3_32b_fs_ctxwide_noleg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_FS_CTX2t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN3_32B_FS_CTX2t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n ${JOB}_qwen3_32b_fs_ctxwide_leg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_FS_CTX2t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN3_32B_ZScot_CTX2t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n ${JOB}_qwen3_32b_zscot_ctxwide_noleg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_ZScot_CTX2t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN3_32B_ZScot_CTX2t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n ${JOB}_qwen3_32b_zscot_ctxwide_leg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_ZScot_CTX2t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN3_32B_FScot_CTX2t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n ${JOB}_qwen3_32b_fscot_ctxwide_noleg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_FScot_CTX2t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN3_32B_FScot_CTX2t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n ${JOB}_qwen3_32b_fscot_ctxwide_leg \
  -o Decoder-Only/Qwen3-32B/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model Qwen/Qwen3-32B \
    --backend qwen3 \
    --sheet QWEN3_32B_FScot_CTX2t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Qwen3-32B/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

fi

# ============================================================
# QWEN3_32B_THINKING — Qwen/Qwen3-32B (thinking/reasoning mode)
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "qwen3_32bthinking" ]]; then

mkdir -p Decoder-Only/Qwen3-32B-Thinking/slurm_logs
mkdir -p Decoder-Only/Qwen3-32B-Thinking/data

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN3_32BT_ZS_CTX2t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n ${JOB}_qwen3_32bt_zs_ctxwide_noleg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32BT_ZS_CTX2t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN3_32BT_ZS_CTX2t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n ${JOB}_qwen3_32bt_zs_ctxwide_leg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32BT_ZS_CTX2t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN3_32BT_FS_CTX2t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n ${JOB}_qwen3_32bt_fs_ctxwide_noleg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32BT_FS_CTX2t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN3_32BT_FS_CTX2t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n ${JOB}_qwen3_32bt_fs_ctxwide_leg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32BT_FS_CTX2t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN3_32BT_ZScot_CTX2t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n ${JOB}_qwen3_32bt_zscot_ctxwide_noleg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32BT_ZScot_CTX2t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN3_32BT_ZScot_CTX2t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n ${JOB}_qwen3_32bt_zscot_ctxwide_leg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32BT_ZScot_CTX2t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN3_32BT_FScot_CTX2t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n ${JOB}_qwen3_32bt_fscot_ctxwide_noleg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32BT_FScot_CTX2t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN3_32BT_FScot_CTX2t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n ${JOB}_qwen3_32bt_fscot_ctxwide_leg \
  -o Decoder-Only/Qwen3-32B-Thinking/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model Qwen/Qwen3-32B \
    --backend qwen3_thinking \
    --sheet QWEN3_32BT_FScot_CTX2t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/Qwen3-32B-Thinking/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

fi

# ============================================================
# O4MINI — o4-mini (OpenAI reasoning model)
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "o4mini" ]]; then

mkdir -p Decoder-Only/O4-Mini/slurm_logs
mkdir -p Decoder-Only/O4-Mini/data

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: O4MINI_ZS_CTX2t_noLeg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_o4mini_zs_ctxwide_noleg \
  -o Decoder-Only/O4-Mini/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model o4-mini \
    --backend openai_reasoning \
    --reasoning_effort medium \
    --sheet O4MINI_ZS_CTX2t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/O4-Mini/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: O4MINI_ZS_CTX2t_Leg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_o4mini_zs_ctxwide_leg \
  -o Decoder-Only/O4-Mini/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model o4-mini \
    --backend openai_reasoning \
    --reasoning_effort medium \
    --sheet O4MINI_ZS_CTX2t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/O4-Mini/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: O4MINI_FS_CTX2t_noLeg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_o4mini_fs_ctxwide_noleg \
  -o Decoder-Only/O4-Mini/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model o4-mini \
    --backend openai_reasoning \
    --reasoning_effort medium \
    --sheet O4MINI_FS_CTX2t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/O4-Mini/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: O4MINI_FS_CTX2t_Leg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_o4mini_fs_ctxwide_leg \
  -o Decoder-Only/O4-Mini/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model o4-mini \
    --backend openai_reasoning \
    --reasoning_effort medium \
    --sheet O4MINI_FS_CTX2t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/O4-Mini/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: O4MINI_ZScot_CTX2t_noLeg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_o4mini_zscot_ctxwide_noleg \
  -o Decoder-Only/O4-Mini/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model o4-mini \
    --backend openai_reasoning \
    --reasoning_effort medium \
    --sheet O4MINI_ZScot_CTX2t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/O4-Mini/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: O4MINI_ZScot_CTX2t_Leg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_o4mini_zscot_ctxwide_leg \
  -o Decoder-Only/O4-Mini/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model o4-mini \
    --backend openai_reasoning \
    --reasoning_effort medium \
    --sheet O4MINI_ZScot_CTX2t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/O4-Mini/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: O4MINI_FScot_CTX2t_noLeg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_o4mini_fscot_ctxwide_noleg \
  -o Decoder-Only/O4-Mini/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model o4-mini \
    --backend openai_reasoning \
    --reasoning_effort medium \
    --sheet O4MINI_FScot_CTX2t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/O4-Mini/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: O4MINI_FScot_CTX2t_Leg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_o4mini_fscot_ctxwide_leg \
  -o Decoder-Only/O4-Mini/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model o4-mini \
    --backend openai_reasoning \
    --reasoning_effort medium \
    --sheet O4MINI_FScot_CTX2t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/O4-Mini/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

fi

# ============================================================
# O3 — o3 (OpenAI flagship reasoning model)
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "o3" ]]; then

mkdir -p Decoder-Only/O3/slurm_logs
mkdir -p Decoder-Only/O3/data

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: O3_ZS_CTX2t_noLeg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_o3_zs_ctxwide_noleg \
  -o Decoder-Only/O3/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model o3 \
    --backend openai_reasoning \
    --reasoning_effort high \
    --sheet O3_ZS_CTX2t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/O3/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: O3_ZS_CTX2t_Leg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_o3_zs_ctxwide_leg \
  -o Decoder-Only/O3/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model o3 \
    --backend openai_reasoning \
    --reasoning_effort high \
    --sheet O3_ZS_CTX2t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/O3/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: O3_FS_CTX2t_noLeg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_o3_fs_ctxwide_noleg \
  -o Decoder-Only/O3/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model o3 \
    --backend openai_reasoning \
    --reasoning_effort high \
    --sheet O3_FS_CTX2t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/O3/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: O3_FS_CTX2t_Leg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_o3_fs_ctxwide_leg \
  -o Decoder-Only/O3/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model o3 \
    --backend openai_reasoning \
    --reasoning_effort high \
    --sheet O3_FS_CTX2t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/O3/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: O3_ZScot_CTX2t_noLeg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_o3_zscot_ctxwide_noleg \
  -o Decoder-Only/O3/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model o3 \
    --backend openai_reasoning \
    --reasoning_effort high \
    --sheet O3_ZScot_CTX2t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/O3/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: O3_ZScot_CTX2t_Leg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_o3_zscot_ctxwide_leg \
  -o Decoder-Only/O3/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model o3 \
    --backend openai_reasoning \
    --reasoning_effort high \
    --sheet O3_ZScot_CTX2t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/O3/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: O3_FScot_CTX2t_noLeg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_o3_fscot_ctxwide_noleg \
  -o Decoder-Only/O3/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model o3 \
    --backend openai_reasoning \
    --reasoning_effort high \
    --sheet O3_FScot_CTX2t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/O3/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: O3_FScot_CTX2t_Leg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_o3_fscot_ctxwide_leg \
  -o Decoder-Only/O3/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model o3 \
    --backend openai_reasoning \
    --reasoning_effort high \
    --sheet O3_FScot_CTX2t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/O3/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

fi

# ============================================================
# O3MINI — o3-mini (OpenAI compact reasoning model)
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "o3mini" ]]; then

mkdir -p Decoder-Only/O3-Mini/slurm_logs
mkdir -p Decoder-Only/O3-Mini/data

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: O3MINI_ZS_CTX2t_noLeg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_o3mini_zs_ctxwide_noleg \
  -o Decoder-Only/O3-Mini/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model o3-mini \
    --backend openai_reasoning \
    --reasoning_effort medium \
    --sheet O3MINI_ZS_CTX2t_noLeg \
    --instruction_type zero_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/O3-Mini/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: O3MINI_ZS_CTX2t_Leg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_o3mini_zs_ctxwide_leg \
  -o Decoder-Only/O3-Mini/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model o3-mini \
    --backend openai_reasoning \
    --reasoning_effort medium \
    --sheet O3MINI_ZS_CTX2t_Leg \
    --instruction_type zero_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/O3-Mini/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: O3MINI_FS_CTX2t_noLeg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_o3mini_fs_ctxwide_noleg \
  -o Decoder-Only/O3-Mini/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model o3-mini \
    --backend openai_reasoning \
    --reasoning_effort medium \
    --sheet O3MINI_FS_CTX2t_noLeg \
    --instruction_type few_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/O3-Mini/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: O3MINI_FS_CTX2t_Leg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_o3mini_fs_ctxwide_leg \
  -o Decoder-Only/O3-Mini/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model o3-mini \
    --backend openai_reasoning \
    --reasoning_effort medium \
    --sheet O3MINI_FS_CTX2t_Leg \
    --instruction_type few_shot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/O3-Mini/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: O3MINI_ZScot_CTX2t_noLeg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_o3mini_zscot_ctxwide_noleg \
  -o Decoder-Only/O3-Mini/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model o3-mini \
    --backend openai_reasoning \
    --reasoning_effort medium \
    --sheet O3MINI_ZScot_CTX2t_noLeg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/O3-Mini/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: O3MINI_ZScot_CTX2t_Leg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_o3mini_zscot_ctxwide_leg \
  -o Decoder-Only/O3-Mini/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model o3-mini \
    --backend openai_reasoning \
    --reasoning_effort medium \
    --sheet O3MINI_ZScot_CTX2t_Leg \
    --instruction_type zero_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/O3-Mini/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: O3MINI_FScot_CTX2t_noLeg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_o3mini_fscot_ctxwide_noleg \
  -o Decoder-Only/O3-Mini/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model o3-mini \
    --backend openai_reasoning \
    --reasoning_effort medium \
    --sheet O3MINI_FScot_CTX2t_noLeg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/O3-Mini/data \
    --dump_prompt \
    --context \
    --context_mode wide"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: O3MINI_FScot_CTX2t_Leg"
nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
  -n ${JOB}_o3mini_fscot_ctxwide_leg \
  -o Decoder-Only/O3-Mini/slurm_logs/%x.out \
  "cd ${BASE_DIR} && \
   ${CONDA_INIT} && \
   conda activate ${CONDA_ENV} && \
   python Decoder-Only/multi_prompt_configs.py \
    --file ${INPUT_FILE} \
    --gold ${GOLD_FILE} \
    --model o3-mini \
    --backend openai_reasoning \
    --reasoning_effort medium \
    --sheet O3MINI_FScot_CTX2t_Leg \
    --instruction_type few_shot_cot \
    --extended \
    --output_format ${OUTPUT_FORMAT} \
    --output_dir Decoder-Only/O3-Mini/data \
    --dump_prompt \
    --context \
    --context_mode wide \
    --dialect_legitimacy"

fi

# ============================================================
# GPT52 — gpt-5.2 (OpenAI non-reasoning, chat model)
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "gpt52" ]]; then

mkdir -p Decoder-Only/GPT52/slurm_logs
mkdir -p Decoder-Only/GPT52/data

for INSTR in zero_shot few_shot zero_shot_cot few_shot_cot; do
  for LEG in "" "--dialect_legitimacy"; do
    [[ -n "$LEG" ]] && LEG_TAG="Leg" || LEG_TAG="noLeg"
    case $INSTR in
      zero_shot)     TAG="ZS"    ;;
      few_shot)      TAG="FS"    ;;
      zero_shot_cot) TAG="ZScot" ;;
      few_shot_cot)  TAG="FScot" ;;
    esac
    SHEET="GPT52_${TAG}_CTX2t_${LEG_TAG}"
    JOB=$((JOB+1))
    echo "[$(printf '%03d' $JOB)] Launching: ${SHEET}"
    nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
      -n ${JOB}_gpt52_${TAG,,}_ctxwide_${LEG_TAG,,} \
      -o Decoder-Only/GPT52/slurm_logs/%x.out \
      "cd ${BASE_DIR} && \
       ${CONDA_INIT} && \
       conda activate ${CONDA_ENV} && \
       python Decoder-Only/multi_prompt_configs.py \
        --file ${INPUT_FILE} \
        --gold ${GOLD_FILE} \
        --model gpt-5.2 \
        --backend openai \
        --sheet ${SHEET} \
        --instruction_type ${INSTR} \
        --extended \
        --output_format ${OUTPUT_FORMAT} \
        --output_dir Decoder-Only/GPT52/data \
        --dump_prompt \
        --context \
        --context_mode wide ${LEG}"
  done
done

fi

# ============================================================
# GEMINI3_FLASH_NOTHINK — gemini-3-flash, thinking disabled
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "gemini3_flash_nothink" ]]; then

mkdir -p Decoder-Only/Gemini3-Flash/slurm_logs
mkdir -p Decoder-Only/Gemini3-Flash/data

for INSTR in zero_shot few_shot zero_shot_cot few_shot_cot; do
  for LEG in "" "--dialect_legitimacy"; do
    [[ -n "$LEG" ]] && LEG_TAG="Leg" || LEG_TAG="noLeg"
    case $INSTR in
      zero_shot)     TAG="ZS"    ;;
      few_shot)      TAG="FS"    ;;
      zero_shot_cot) TAG="ZScot" ;;
      few_shot_cot)  TAG="FScot" ;;
    esac
    SHEET="G3FLASH_${TAG}_CTX2t_${LEG_TAG}"
    JOB=$((JOB+1))
    echo "[$(printf '%03d' $JOB)] Launching: ${SHEET}"
    nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
      -n ${JOB}_g3flash_${TAG,,}_ctxwide_${LEG_TAG,,} \
      -o Decoder-Only/Gemini3-Flash/slurm_logs/%x.out \
      "cd ${BASE_DIR} && \
       ${CONDA_INIT} && \
       conda activate ${CONDA_ENV} && \
       python Decoder-Only/multi_prompt_configs.py \
        --file ${INPUT_FILE} \
        --gold ${GOLD_FILE} \
        --model gemini-3-flash-preview \
        --backend gemini3 \
        --thinking_level minimal \
        --sheet ${SHEET} \
        --instruction_type ${INSTR} \
        --extended \
        --output_format ${OUTPUT_FORMAT} \
        --output_dir Decoder-Only/Gemini3-Flash/data \
        --dump_prompt \
        --context \
        --context_mode wide ${LEG}"
  done
done

fi

# ============================================================
# GEMINI3_FLASH — gemini-3-flash, thinking enabled (high)
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "gemini3_flash" ]]; then

mkdir -p Decoder-Only/Gemini3-Flash-Thinking/slurm_logs
mkdir -p Decoder-Only/Gemini3-Flash-Thinking/data

for INSTR in zero_shot few_shot zero_shot_cot few_shot_cot; do
  for LEG in "" "--dialect_legitimacy"; do
    [[ -n "$LEG" ]] && LEG_TAG="Leg" || LEG_TAG="noLeg"
    case $INSTR in
      zero_shot)     TAG="ZS"    ;;
      few_shot)      TAG="FS"    ;;
      zero_shot_cot) TAG="ZScot" ;;
      few_shot_cot)  TAG="FScot" ;;
    esac
    SHEET="G3FLASHT_${TAG}_CTX2t_${LEG_TAG}"
    JOB=$((JOB+1))
    echo "[$(printf '%03d' $JOB)] Launching: ${SHEET}"
    nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
      -n ${JOB}_g3flasht_${TAG,,}_ctxwide_${LEG_TAG,,} \
      -o Decoder-Only/Gemini3-Flash-Thinking/slurm_logs/%x.out \
      "cd ${BASE_DIR} && \
       ${CONDA_INIT} && \
       conda activate ${CONDA_ENV} && \
       python Decoder-Only/multi_prompt_configs.py \
        --file ${INPUT_FILE} \
        --gold ${GOLD_FILE} \
        --model gemini-3-flash-preview \
        --backend gemini3 \
        --thinking_level high \
        --sheet ${SHEET} \
        --instruction_type ${INSTR} \
        --extended \
        --output_format ${OUTPUT_FORMAT} \
        --output_dir Decoder-Only/Gemini3-Flash-Thinking/data \
        --dump_prompt \
        --context \
        --context_mode wide ${LEG}"
  done
done

fi

# ============================================================
# GEMINI25_PRO — gemini-2.5-pro (reasoning/thinking)
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "gemini25_pro" ]]; then

mkdir -p Decoder-Only/Gemini25-Pro/slurm_logs
mkdir -p Decoder-Only/Gemini25-Pro/data

for INSTR in zero_shot few_shot zero_shot_cot few_shot_cot; do
  for LEG in "" "--dialect_legitimacy"; do
    [[ -n "$LEG" ]] && LEG_TAG="Leg" || LEG_TAG="noLeg"
    case $INSTR in
      zero_shot)     TAG="ZS"    ;;
      few_shot)      TAG="FS"    ;;
      zero_shot_cot) TAG="ZScot" ;;
      few_shot_cot)  TAG="FScot" ;;
    esac
    SHEET="GEMINI25P_${TAG}_CTX2t_${LEG_TAG}"
    JOB=$((JOB+1))
    echo "[$(printf '%03d' $JOB)] Launching: ${SHEET}"
    nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
      -n ${JOB}_gemini25p_${TAG,,}_ctxwide_${LEG_TAG,,} \
      -o Decoder-Only/Gemini25-Pro/slurm_logs/%x.out \
      "cd ${BASE_DIR} && \
       ${CONDA_INIT} && \
       conda activate ${CONDA_ENV} && \
       python Decoder-Only/multi_prompt_configs.py \
        --file ${INPUT_FILE} \
        --gold ${GOLD_FILE} \
        --model gemini-2.5-pro \
        --backend gemini3 \
        --thinking_level high \
        --sheet ${SHEET} \
        --instruction_type ${INSTR} \
        --extended \
        --output_format ${OUTPUT_FORMAT} \
        --output_dir Decoder-Only/Gemini25-Pro/data \
        --dump_prompt \
        --context \
        --context_mode wide ${LEG}"
  done
done

fi

# ============================================================
# GPT5 — gpt-5 (OpenAI reasoning)
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "gpt5" ]]; then

mkdir -p Decoder-Only/GPT5/slurm_logs
mkdir -p Decoder-Only/GPT5/data

for INSTR in zero_shot few_shot zero_shot_cot few_shot_cot; do
  for LEG in "" "--dialect_legitimacy"; do
    [[ -n "$LEG" ]] && LEG_TAG="Leg" || LEG_TAG="noLeg"
    case $INSTR in
      zero_shot)     TAG="ZS"    ;;
      few_shot)      TAG="FS"    ;;
      zero_shot_cot) TAG="ZScot" ;;
      few_shot_cot)  TAG="FScot" ;;
    esac
    SHEET="GPT5_${TAG}_CTX2t_${LEG_TAG}"
    JOB=$((JOB+1))
    echo "[$(printf '%03d' $JOB)] Launching: ${SHEET}"
    nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
      -n ${JOB}_gpt5_${TAG,,}_ctxwide_${LEG_TAG,,} \
      -o Decoder-Only/GPT5/slurm_logs/%x.out \
      "cd ${BASE_DIR} && \
       ${CONDA_INIT} && \
       conda activate ${CONDA_ENV} && \
       python Decoder-Only/multi_prompt_configs.py \
        --file ${INPUT_FILE} \
        --gold ${GOLD_FILE} \
        --model gpt-5 \
        --backend openai_reasoning \
        --reasoning_effort high \
        --sheet ${SHEET} \
        --instruction_type ${INSTR} \
        --extended \
        --output_format ${OUTPUT_FORMAT} \
        --output_dir Decoder-Only/GPT5/data \
        --dump_prompt \
        --context \
        --context_mode wide ${LEG}"
  done
done

fi

# ============================================================
# O3DEEP — o3-deep-research (OpenAI, high reasoning effort)
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "o3deep" ]]; then

mkdir -p Decoder-Only/O3-Deep/slurm_logs
mkdir -p Decoder-Only/O3-Deep/data

for INSTR in zero_shot few_shot zero_shot_cot few_shot_cot; do
  for LEG in "" "--dialect_legitimacy"; do
    [[ -n "$LEG" ]] && LEG_TAG="Leg" || LEG_TAG="noLeg"
    case $INSTR in
      zero_shot)     TAG="ZS"    ;;
      few_shot)      TAG="FS"    ;;
      zero_shot_cot) TAG="ZScot" ;;
      few_shot_cot)  TAG="FScot" ;;
    esac
    SHEET="O3DEEP_${TAG}_CTX2t_${LEG_TAG}"
    JOB=$((JOB+1))
    echo "[$(printf '%03d' $JOB)] Launching: ${SHEET}"
    nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
      -n ${JOB}_o3deep_${TAG,,}_ctxwide_${LEG_TAG,,} \
      -o Decoder-Only/O3-Deep/slurm_logs/%x.out \
      "cd ${BASE_DIR} && \
       ${CONDA_INIT} && \
       conda activate ${CONDA_ENV} && \
       python Decoder-Only/multi_prompt_configs.py \
        --file ${INPUT_FILE} \
        --gold ${GOLD_FILE} \
        --model o3-deep-research \
        --backend openai_reasoning \
        --reasoning_effort high \
        --sheet ${SHEET} \
        --instruction_type ${INSTR} \
        --extended \
        --output_format ${OUTPUT_FORMAT} \
        --output_dir Decoder-Only/O3-Deep/data \
        --dump_prompt \
        --context \
        --context_mode wide ${LEG}"
  done
done

fi

# ============================================================
# O4MINIDEEP — o4-mini-deep-research (OpenAI, high reasoning effort)
# ============================================================

if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "o4miniDeep" ]]; then

mkdir -p Decoder-Only/O4-Mini-Deep/slurm_logs
mkdir -p Decoder-Only/O4-Mini-Deep/data

for INSTR in zero_shot few_shot zero_shot_cot few_shot_cot; do
  for LEG in "" "--dialect_legitimacy"; do
    [[ -n "$LEG" ]] && LEG_TAG="Leg" || LEG_TAG="noLeg"
    case $INSTR in
      zero_shot)     TAG="ZS"    ;;
      few_shot)      TAG="FS"    ;;
      zero_shot_cot) TAG="ZScot" ;;
      few_shot_cot)  TAG="FScot" ;;
    esac
    SHEET="O4MINID_${TAG}_CTX2t_${LEG_TAG}"
    JOB=$((JOB+1))
    echo "[$(printf '%03d' $JOB)] Launching: ${SHEET}"
    nlprun -q ${API_QUEUE} -p standard -r 40G -c 2 \
      -n ${JOB}_o4minid_${TAG,,}_ctxwide_${LEG_TAG,,} \
      -o Decoder-Only/O4-Mini-Deep/slurm_logs/%x.out \
      "cd ${BASE_DIR} && \
       ${CONDA_INIT} && \
       conda activate ${CONDA_ENV} && \
       python Decoder-Only/multi_prompt_configs.py \
        --file ${INPUT_FILE} \
        --gold ${GOLD_FILE} \
        --model o4-mini-deep-research \
        --backend openai_reasoning \
        --reasoning_effort high \
        --sheet ${SHEET} \
        --instruction_type ${INSTR} \
        --extended \
        --output_format ${OUTPUT_FORMAT} \
        --output_dir Decoder-Only/O4-Mini-Deep/data \
        --dump_prompt \
        --context \
        --context_mode wide ${LEG}"
  done
done

fi

echo "Done. Submitted ${JOB} CTX2t re-run jobs for $MODEL_FILTER."
