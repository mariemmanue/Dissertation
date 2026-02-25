#!/bin/bash
# Re-run all CTX2t (two_turn context) experiments after the assistant-ack fix.
#
# Models: gemini, phi4, qwen25_7b, gpt41, qwen3_32b
# Grid:   4 instruction_types × 1 context(two_turn) × 2 legitimacy = 8 per model
# Total:  40 jobs
#
# Usage:
#   chmod +x Decoder-Only/rerun_ctx2t.sh
#   ./Decoder-Only/rerun_ctx2t.sh              # launch all 40 jobs
#   ./Decoder-Only/rerun_ctx2t.sh phi4         # launch only phi4 CTX2t jobs (8)
#   ./Decoder-Only/rerun_ctx2t.sh gemini       # launch only gemini CTX2t jobs (8)

set -e

MODEL_FILTER="${1:-all}"

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
  -n ${JOB}_phi4_zs_ctx2t_noleg \
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
    --context_mode two_turn"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: PHI4_ZS_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_phi4_zs_ctx2t_leg \
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
    --context_mode two_turn \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: PHI4_FS_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_phi4_fs_ctx2t_noleg \
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
    --context_mode two_turn"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: PHI4_FS_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_phi4_fs_ctx2t_leg \
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
    --context_mode two_turn \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: PHI4_ZScot_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_phi4_zscot_ctx2t_noleg \
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
    --context_mode two_turn"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: PHI4_ZScot_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_phi4_zscot_ctx2t_leg \
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
    --context_mode two_turn \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: PHI4_FScot_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_phi4_fscot_ctx2t_noleg \
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
    --context_mode two_turn"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: PHI4_FScot_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_phi4_fscot_ctx2t_leg \
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
    --context_mode two_turn \
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
nlprun -q jag -p standard -r 40G -c 2 \
  -n ${JOB}_gemini_zs_ctx2t_noleg \
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
    --context_mode two_turn"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GEMINI_ZS_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n ${JOB}_gemini_zs_ctx2t_leg \
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
    --context_mode two_turn \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GEMINI_FS_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n ${JOB}_gemini_fs_ctx2t_noleg \
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
    --context_mode two_turn"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GEMINI_FS_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n ${JOB}_gemini_fs_ctx2t_leg \
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
    --context_mode two_turn \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GEMINI_ZScot_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n ${JOB}_gemini_zscot_ctx2t_noleg \
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
    --context_mode two_turn"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GEMINI_ZScot_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n ${JOB}_gemini_zscot_ctx2t_leg \
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
    --context_mode two_turn \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GEMINI_FScot_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n ${JOB}_gemini_fscot_ctx2t_noleg \
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
    --context_mode two_turn"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GEMINI_FScot_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n ${JOB}_gemini_fscot_ctx2t_leg \
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
    --context_mode two_turn \
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
  -n ${JOB}_qwen25_7b_zs_ctx2t_noleg \
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
    --context_mode two_turn"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN25_7B_ZS_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_qwen25_7b_zs_ctx2t_leg \
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
    --context_mode two_turn \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN25_7B_FS_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_qwen25_7b_fs_ctx2t_noleg \
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
    --context_mode two_turn"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN25_7B_FS_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_qwen25_7b_fs_ctx2t_leg \
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
    --context_mode two_turn \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN25_7B_ZScot_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_qwen25_7b_zscot_ctx2t_noleg \
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
    --context_mode two_turn"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN25_7B_ZScot_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_qwen25_7b_zscot_ctx2t_leg \
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
    --context_mode two_turn \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN25_7B_FScot_CTX2t_noLeg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_qwen25_7b_fscot_ctx2t_noleg \
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
    --context_mode two_turn"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN25_7B_FScot_CTX2t_Leg"
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n ${JOB}_qwen25_7b_fscot_ctx2t_leg \
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
    --context_mode two_turn \
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
nlprun -q jag -p standard -r 40G -c 2 \
  -n ${JOB}_gpt41_zs_ctx2t_noleg \
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
    --context_mode two_turn"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GPT41_ZS_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n ${JOB}_gpt41_zs_ctx2t_leg \
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
    --context_mode two_turn \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GPT41_FS_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n ${JOB}_gpt41_fs_ctx2t_noleg \
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
    --context_mode two_turn"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GPT41_FS_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n ${JOB}_gpt41_fs_ctx2t_leg \
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
    --context_mode two_turn \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GPT41_ZScot_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n ${JOB}_gpt41_zscot_ctx2t_noleg \
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
    --context_mode two_turn"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GPT41_ZScot_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n ${JOB}_gpt41_zscot_ctx2t_leg \
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
    --context_mode two_turn \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GPT41_FScot_CTX2t_noLeg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n ${JOB}_gpt41_fscot_ctx2t_noleg \
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
    --context_mode two_turn"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: GPT41_FScot_CTX2t_Leg"
nlprun -q jag -p standard -r 40G -c 2 \
  -n ${JOB}_gpt41_fscot_ctx2t_leg \
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
    --context_mode two_turn \
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
  -n ${JOB}_qwen3_32b_zs_ctx2t_noleg \
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
    --context_mode two_turn"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN3_32B_ZS_CTX2t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n ${JOB}_qwen3_32b_zs_ctx2t_leg \
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
    --context_mode two_turn \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN3_32B_FS_CTX2t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n ${JOB}_qwen3_32b_fs_ctx2t_noleg \
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
    --context_mode two_turn"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN3_32B_FS_CTX2t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n ${JOB}_qwen3_32b_fs_ctx2t_leg \
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
    --context_mode two_turn \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN3_32B_ZScot_CTX2t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n ${JOB}_qwen3_32b_zscot_ctx2t_noleg \
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
    --context_mode two_turn"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN3_32B_ZScot_CTX2t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n ${JOB}_qwen3_32b_zscot_ctx2t_leg \
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
    --context_mode two_turn \
    --dialect_legitimacy"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN3_32B_FScot_CTX2t_noLeg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n ${JOB}_qwen3_32b_fscot_ctx2t_noleg \
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
    --context_mode two_turn"

JOB=$((JOB+1))
echo "[$(printf '%03d' $JOB)] Launching: QWEN3_32B_FScot_CTX2t_Leg"
nlprun -g 2 -q sphinx -p standard -r 200G -c 4 \
  -n ${JOB}_qwen3_32b_fscot_ctx2t_leg \
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
    --context_mode two_turn \
    --dialect_legitimacy"

fi

echo "Done. Submitted ${JOB} CTX2t re-run jobs for $MODEL_FILTER."
