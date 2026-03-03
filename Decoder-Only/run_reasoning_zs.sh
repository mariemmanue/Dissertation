#!/bin/bash
# Zero-shot (non-CoT) experiments for all reasoning models,
# varied across context mode and dialect legitimacy.
#
# Reasoning models: qwen3_32bthinking, gemini3_pro, gemini3_flash, gemini25_pro,
#                   gpt5, o4mini, o3, o3mini, o3deep, o4miniDeep
# Grid:  1 instruction_type (zero_shot)
#       × 3 context conditions (noCtx, CTXsingle, CTXwide)
#       × 2 legitimacy (noLeg, Leg)
#       = 6 jobs per model
# Total: 60 jobs
#
# Usage:
#   chmod +x Decoder-Only/run_reasoning_zs.sh
#   ./Decoder-Only/run_reasoning_zs.sh                # launch all 60 jobs
#   ./Decoder-Only/run_reasoning_zs.sh o3             # launch only o3 ZS jobs (6)
#   ./Decoder-Only/run_reasoning_zs.sh gemini3_pro    # launch only Gemini 3 Pro ZS jobs (6)

set -e

MODEL_FILTER="${1:-all}"

BASE_DIR="/nlp/scr/mtano/Dissertation"
CONDA_INIT=". /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="cgedit"
INPUT_FILE="Datasets/FullTest_Final.xlsx"
GOLD_FILE="Datasets/FullTest_Final.xlsx"
OUTPUT_FORMAT="markdown"

JOB=0

# ----------------------------------------------------------------
# Helper: launch 6 ZS jobs (3 context modes × 2 legitimacy) for one model.
#
# Args:
#   $1  filter_name    — MODEL_FILTER key (e.g. "o3")
#   $2  model_id       — model passed to --model
#   $3  backend        — backend passed to --backend
#   $4  sheet_prefix   — sheet name prefix (e.g. "O3")
#   $5  outdir         — output dir (e.g. "Decoder-Only/O3")
#   $6  nlprun_res     — nlprun resource flags (word-splits intentionally)
#   $7  extra_args     — extra python args (e.g. "--reasoning_effort high")
# ----------------------------------------------------------------
launch_zs() {
  local NAME="$1" MODEL="$2" BACKEND="$3" PREFIX="$4" OUTDIR="$5"
  local RES="$6"
  local EXTRA="$7"

  [[ "$MODEL_FILTER" != "all" && "$MODEL_FILTER" != "$NAME" ]] && return

  mkdir -p "${OUTDIR}/slurm_logs"
  mkdir -p "${OUTDIR}/data"

  # Context conditions: tag, --context flag, --context_mode value
  # Format: "CTX_TAG|CTX_FLAG|CTX_MODE"
  #   noCtx     → no --context flag, no --context_mode
  #   CTXsingle → --context --context_mode single_turn
  #   CTXwide   → --context --context_mode wide
  local -a CTX_CONDITIONS=("noCtx||" "CTXsingle|--context|single_turn" "CTXwide|--context|wide")

  for CTX_DEF in "${CTX_CONDITIONS[@]}"; do
    IFS='|' read -r CTX_TAG CTX_FLAG CTX_MODE <<< "$CTX_DEF"
    [[ -n "$CTX_MODE" ]] && CTX_ARGS="${CTX_FLAG} --context_mode ${CTX_MODE}" || CTX_ARGS=""

    for LEG in "" "--dialect_legitimacy"; do
      [[ -n "$LEG" ]] && LEG_TAG="Leg" || LEG_TAG="noLeg"
      SHEET="${PREFIX}_ZS_${CTX_TAG}_${LEG_TAG}"
      JOB=$((JOB+1))
      echo "[$(printf '%03d' $JOB)] Launching: ${SHEET}"
      # shellcheck disable=SC2086  # intentional word-split on $RES / $CTX_ARGS / $LEG
      nlprun $RES \
        -n ${JOB}_${NAME}_zs_${CTX_TAG,,}_${LEG_TAG,,} \
        -o "${OUTDIR}/slurm_logs/%x.out" \
        "cd ${BASE_DIR} && \
         ${CONDA_INIT} && \
         conda activate ${CONDA_ENV} && \
         python Decoder-Only/multi_prompt_configs.py \
          --file ${INPUT_FILE} \
          --gold ${GOLD_FILE} \
          --model ${MODEL} \
          --backend ${BACKEND} \
          ${EXTRA} \
          --sheet ${SHEET} \
          --instruction_type zero_shot \
          --extended \
          --output_format ${OUTPUT_FORMAT} \
          --output_dir ${OUTDIR}/data \
          --dump_prompt \
          ${CTX_ARGS} ${LEG}"
    done
  done
}

# ============================================================
# REASONING MODELS
# ============================================================
#
#  Name               Model ID                  Backend            Sheet prefix  Output dir                           Resources                                Extra args
# -------            ---------                  -------            ------------  ----------                           ---------                                ----------
launch_zs qwen3_32bthinking  Qwen/Qwen3-32B              qwen3_thinking   QWEN3_32BT   Decoder-Only/Qwen3-32B-Thinking      "-g 2 -q sphinx -p standard -r 200G -c 4"  ""
launch_zs gemini3_pro        gemini-3-pro-preview        gemini3          GEMINI3_PRO  Decoder-Only/Gemini3_Pro             "-q jag -p standard -r 40G -c 2"           "--thinking_level high"
launch_zs gemini3_flash      gemini-3-flash              gemini3          G3FLASHT     Decoder-Only/Gemini3-Flash-Thinking  "-q jag -p standard -r 40G -c 2"           "--thinking_level high"
launch_zs gemini25_pro       gemini-2.5-pro-preview      gemini3          GEMINI25P    Decoder-Only/Gemini25-Pro            "-q jag -p standard -r 40G -c 2"           "--thinking_level high"
launch_zs gpt5               gpt-5                       openai_reasoning GPT5         Decoder-Only/GPT5                    "-q jag -p standard -r 40G -c 2"           "--reasoning_effort high"
launch_zs o4mini             o4-mini                     openai_reasoning O4MINI       Decoder-Only/O4-Mini                 "-q jag -p standard -r 40G -c 2"           "--reasoning_effort medium"
launch_zs o3                 o3                          openai_reasoning O3           Decoder-Only/O3                      "-q jag -p standard -r 40G -c 2"           "--reasoning_effort high"
launch_zs o3mini             o3-mini                     openai_reasoning O3MINI       Decoder-Only/O3-Mini                 "-q jag -p standard -r 40G -c 2"           "--reasoning_effort medium"
launch_zs o3deep             o3-deep-research            openai_reasoning O3DEEP       Decoder-Only/O3-Deep                 "-q jag -p standard -r 40G -c 2"           "--reasoning_effort high"
launch_zs o4miniDeep         o4-mini-deep-research       openai_reasoning O4MINID      Decoder-Only/O4-Mini-Deep            "-q jag -p standard -r 40G -c 2"           "--reasoning_effort high"

echo "Done. Submitted ${JOB} reasoning ZS jobs for ${MODEL_FILTER}."
