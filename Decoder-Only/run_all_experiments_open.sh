#!/bin/bash
# Open-source model experiment runner
# Total jobs: 168 (7 models × 24 configs each)
#
# Grid per model:
#   instruction_type:   zero_shot, few_shot, zero_shot_cot, few_shot_cot  (4)
#   context:            noCTX, CTX1t, CTX5                                (3)
#   dialect_legitimacy: off, on                                           (2)
#   output_format:      json (all models)
#   ──────────────────────────────────────────────────────────────────────
#   Total per model:    24
#
# Usage:
#   chmod +x run_all_experiments_open.sh
#   ./run_all_experiments_open.sh                   # all 168 jobs
#   ./run_all_experiments_open.sh phi4              # only phi4 (24 jobs)
#   ./run_all_experiments_open.sh phi4_reasoning    # only Phi-4-reasoning (24 jobs)
#   ./run_all_experiments_open.sh llama             # only Llama-3.1-70B (24 jobs)
#   ./run_all_experiments_open.sh qwen25            # only Qwen2.5-7B (24 jobs)
#   ./run_all_experiments_open.sh qwen3             # only Qwen3-32B no-think (24 jobs)
#   ./run_all_experiments_open.sh qwen3_thinking    # only Qwen3-32B thinking (24 jobs)
#   ./run_all_experiments_open.sh qwq               # only QwQ-32B (24 jobs)
#
# Reasoning models (phi4_reasoning, qwen3_thinking, qwq) get -r 120G.
# All models use sphinx GPU queue.

set -e

MODEL_FILTER="${1:-all}"

BASE="cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/open_source_configs.py"

FILE="--file Datasets/FullTest_Final.xlsx --gold Datasets/FullTest_Final.xlsx"

# ─────────────────────────────────────────────────────────────────────────────
# Helper: emit one nlprun job
#   run_job JOB_NUM JOB_NAME SLURM_LOG_DIR MEM MODEL_STR BACKEND SHEET_PREFIX \
#           INSTRUCTION_TYPE CONTEXT_FLAG CONTEXT_MODE_FLAG LEG_FLAG OUTPUT_DIR
# ─────────────────────────────────────────────────────────────────────────────
run_job() {
    local num="$1"
    local name="$2"
    local logdir="$3"
    local mem="$4"
    local model="$5"
    local backend="$6"
    local sheet="$7"
    local itype="$8"
    local ctx_flag="$9"
    local leg_flag="${10}"
    local outdir="${11}"

    printf "[%03d] Launching: %s\n" "$num" "$sheet"
    nlprun -g 1 -q sphinx -p standard -r "${mem}" -c 4 \
      -n "${num}_${name}" \
      -o "Decoder-Only/${logdir}/%x.out" \
      "${BASE} \
        ${FILE} \
        --model ${model} \
        --backend ${backend} \
        --sheet ${sheet} \
        --instruction_type ${itype} \
        --extended \
        --output_format json \
        --output_dir Decoder-Only/${outdir} \
        --dump_prompt \
        ${ctx_flag} \
        ${leg_flag}"
}

# ─────────────────────────────────────────────────────────────────────────────
# emit_model_jobs: emit all 24 jobs for one model
#   emit_model_jobs START_JOB SHORT_KEY MEM MODEL BACKEND SHEET_PFX OUTDIR LOGDIR
# ─────────────────────────────────────────────────────────────────────────────
emit_model_jobs() {
    local start="$1"
    local key="$2"
    local mem="$3"
    local model="$4"
    local backend="$5"
    local spfx="$6"    # sheet prefix e.g. PHI4
    local outdir="$7"
    local logdir="$8"

    mkdir -p "Decoder-Only/${logdir}"
    mkdir -p "Decoder-Only/${outdir}"

    local n=$start
    for itype in zero_shot few_shot zero_shot_cot few_shot_cot; do
        # Sheet tag
        case "$itype" in
            zero_shot)     itag="ZS"    ;;
            few_shot)      itag="FS"    ;;
            zero_shot_cot) itag="ZSCOT" ;;
            few_shot_cot)  itag="FSCOT" ;;
        esac

        for ctx in noCTX CTX1t CTX5; do
            case "$ctx" in
                noCTX) ctx_flag="" ;;
                CTX1t) ctx_flag="--context --context_mode single_turn" ;;
                CTX5)  ctx_flag="--context --context_mode wide" ;;
            esac

            for leg in noLeg Leg; do
                if [[ "$leg" == "Leg" ]]; then
                    leg_flag="--dialect_legitimacy"
                else
                    leg_flag=""
                fi

                sheet="${spfx}_${itag}_${ctx}_${leg}"
                name="${key}_${itag}_${ctx}_${leg}"
                run_job "$n" "$name" "$logdir" "$mem" \
                        "$model" "$backend" "$sheet" "$itype" \
                        "$ctx_flag" "$leg_flag" "$outdir"
                n=$((n + 1))
            done
        done
    done
}

# ============================================================
# PHI4 — microsoft/phi-4
# ============================================================
if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "phi4" ]]; then
    emit_model_jobs 1 phi4 100G \
        microsoft/phi-4 phi PHI4 \
        Phi-4/data Phi-4/slurm_logs
fi

# ============================================================
# PHI4_REASONING — microsoft/Phi-4-reasoning
# ============================================================
if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "phi4_reasoning" ]]; then
    emit_model_jobs 25 phi4r 120G \
        microsoft/Phi-4-reasoning phi_reasoning PHI4R \
        Phi-4-Reasoning/data Phi-4-Reasoning/slurm_logs
fi

# ============================================================
# LLAMA — meta-llama/Llama-3.1-70B-Instruct
# ============================================================
if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "llama" ]]; then
    emit_model_jobs 49 llama70b 120G \
        meta-llama/Llama-3.1-70B-Instruct llama LLAMA70B \
        Llama-70B/data Llama-70B/slurm_logs
fi

# ============================================================
# QWEN25 — Qwen/Qwen2.5-7B-Instruct
# ============================================================
if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "qwen25" ]]; then
    emit_model_jobs 73 qwen25 100G \
        Qwen/Qwen2.5-7B-Instruct qwen QWEN25 \
        Qwen25/data Qwen25/slurm_logs
fi

# ============================================================
# QWEN3 — Qwen/Qwen3-32B (no thinking)
# ============================================================
if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "qwen3" ]]; then
    emit_model_jobs 97 qwen3nt 100G \
        Qwen/Qwen3-32B qwen3 QWEN3NT \
        Qwen3-NT/data Qwen3-NT/slurm_logs
fi

# ============================================================
# QWEN3_THINKING — Qwen/Qwen3-32B (with thinking)
# ============================================================
if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "qwen3_thinking" ]]; then
    emit_model_jobs 121 qwen3t 120G \
        Qwen/Qwen3-32B qwen3_thinking QWEN3T \
        Qwen3-Think/data Qwen3-Think/slurm_logs
fi

# ============================================================
# QWQ — Qwen/QwQ-32B
# ============================================================
if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "qwq" ]]; then
    emit_model_jobs 145 qwq 120G \
        Qwen/QwQ-32B qwq QWQ \
        QwQ-32B/data QwQ-32B/slurm_logs
fi
