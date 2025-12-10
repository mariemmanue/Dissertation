#!/bin/bash
set -euo pipefail

########################
# MODE: dry | run
# - dry: print commands prefixed with "NLPRUN_CMD:" (no jobs submitted)
# - run: actually call nlprun for every combo

#  MODE=dry bash run_gpt_grid.sh > all_commands.txt
# Inspect a subset (e.g., 25-feature zero-shot, labels_only, no-context):

# grep "GPT_25_ZS" all_commands.txt | grep "_LAB" | grep "_noCTX"

# Actually run that subset:
# grep "GPT_25_ZS" all_commands.txt | grep "_LAB" | grep "_noCTX" \
#   | sed 's/^NLPRUN_CMD: //' | bash

########################
MODE="${MODE:-dry}"   # default to dry for safety

# --------- CONFIGURABLE PATHS & CLUSTER SETTINGS ---------
FILE="data/Test1.xlsx"
OUTDIR="data/results"

QUEUE="jag"
PARTITION="standard"
MEM="40G"
CORES=2
TIME="12:00:00"

# Base working directory on the cluster
WORKDIR="/nlp/scr/mtano/Dissertation/Decoder-Only/GPT"
CONDA_SH="/nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="cgedit"
HF_HOME_DIR="/nlp/scr/mtano/hf_home"

# --------- FACTOR LEVELS ---------
INSTR_TYPES=(zero_shot icl zero_shot_cot few_shot_cot)   # instruction_type
FEATURE_SETS=("17" "25")                                 # 17 = Masis; 25 = extended
LEG_FLAGS=("noLEG" "LEG")                                # dialect_legitimacy off/on
SV_FLAGS=("noSV" "SV")                                   # self_verification off/on
LABEL_FLAGS=("noLAB" "LAB")                              # rationales vs labels_only
CTX_FLAGS=("noCTX" "CTX")                                # without vs with context

# Total combos = 4 * 2 * 2 * 2 * 2 * 2 = 128

# --------- MAIN LOOP ---------
for fs in "${FEATURE_SETS[@]}"; do
  # Extended flag for 25-feature runs
  EXT_FLAG=""
  if [[ "$fs" == "25" ]]; then
    EXT_FLAG="--extended"
  fi

  for instr in "${INSTR_TYPES[@]}"; do
    for leg in "${LEG_FLAGS[@]}"; do
      for sv in "${SV_FLAGS[@]}"; do
        for lab in "${LABEL_FLAGS[@]}"; do
          for ctx in "${CTX_FLAGS[@]}"; do

            # ---- Build short instruction code for sheet name ----
            case "$instr" in
              zero_shot)      INSTR_CODE="ZS"    ;;
              icl)            INSTR_CODE="ICL"   ;;
              zero_shot_cot)  INSTR_CODE="ZSCOT" ;;
              few_shot_cot)   INSTR_CODE="FSCOT" ;;
              *) echo "Unknown instruction_type: $instr"; exit 1 ;;
            esac

            # ---- Base sheet name (without modifiers yet) ----
            SHEET="GPT_${fs}_${INSTR_CODE}"

            # ---- Append LEG/SV/LAB/CTX markers to sheet name ----
            SHEET_FULL="$SHEET"
            if [[ "$leg" == "LEG" ]]; then
              SHEET_FULL+="_LEG"
            else
              SHEET_FULL+="_noLEG"
            fi

            if [[ "$sv" == "SV" ]]; then
              SHEET_FULL+="_SV"
            else
              SHEET_FULL+="_noSV"
            fi

            if [[ "$lab" == "LAB" ]]; then
              SHEET_FULL+="_LAB"
            else
              SHEET_FULL+="_noLAB"
            fi

            if [[ "$ctx" == "CTX" ]]; then
              SHEET_FULL+="_CTX"
            else
              SHEET_FULL+="_noCTX"
            fi

            # ---- Build argument flags ----
            DIALECT_ARGS=""
            if [[ "$leg" == "LEG" ]]; then
              DIALECT_ARGS="--dialect_legitimacy"
            fi

            SV_ARGS=""
            if [[ "$sv" == "SV" ]]; then
              SV_ARGS="--self_verification"
            fi

            LABEL_ARGS=""
            if [[ "$lab" == "LAB" ]]; then
              LABEL_ARGS="--labels_only"
            fi

            CTX_ARGS=""
            if [[ "$ctx" == "CTX" ]]; then
              CTX_ARGS="--context"
            fi

            BLOCK_ARGS=""
            # Only give the model explicit +Example / -Miss lines for ICL & few-shot CoT
            if [[ "$instr" == "icl" || "$instr" == "few_shot_cot" ]]; then
              BLOCK_ARGS="--block_examples"
            fi

            # ---- Job name (lowercase, hyphenated-ish) ----
            JOB_NAME="gpt_${fs}_$(echo "$INSTR_CODE" | tr '[:upper:]' '[:lower:]')"
            if [[ "$leg" == "LEG" ]]; then
              JOB_NAME+="_leg"
            else
              JOB_NAME+="_noleg"
            fi
            if [[ "$sv" == "SV" ]]; then
              JOB_NAME+="_sv"
            else
              JOB_NAME+="_nosv"
            fi
            if [[ "$lab" == "LAB" ]]; then
              JOB_NAME+="_lab"
            else
              JOB_NAME+="_nolab"
            fi
            if [[ "$ctx" == "CTX" ]]; then
              JOB_NAME+="_ctx"
            else
              JOB_NAME+="_noctx"
            fi

            echo "Config: $JOB_NAME (sheet: $SHEET_FULL, instr: $instr, fs: $fs, $leg, $sv, $lab, $ctx)"

            if [[ "$MODE" == "dry" ]]; then
              # Just print the nlprun command, prefixed so we can grep/sed later
              echo "NLPRUN_CMD: nlprun -q $QUEUE -p $PARTITION -r $MEM -c $CORES -t $TIME \
  -n $JOB_NAME \
  -o slurm_logs/%x-%j.out \
  \"cd $WORKDIR && \
   mkdir -p slurm_logs $OUTDIR && \
   . $CONDA_SH && \
   conda activate $CONDA_ENV && \
   export HF_HOME=$HF_HOME_DIR && \
   python gpt_experiments.py \
     --file $FILE \
     --sheet $SHEET_FULL \
     --instruction_type $instr \
     $EXT_FLAG \
     $BLOCK_ARGS \
     $DIALECT_ARGS \
     $SV_ARGS \
     $LABEL_ARGS \
     $CTX_ARGS \
     --output_dir $OUTDIR\""
            else
              # Actually submit job
              nlprun -q "$QUEUE" -p "$PARTITION" -r "$MEM" -c "$CORES" -t "$TIME" \
                -n "$JOB_NAME" \
                -o "slurm_logs/%x-%j.out" \
                "cd $WORKDIR && \
                 mkdir -p slurm_logs $OUTDIR && \
                 . $CONDA_SH && \
                 conda activate $CONDA_ENV && \
                 export HF_HOME=$HF_HOME_DIR && \
                 python gpt_experiments.py \
                   --file $FILE \
                   --sheet $SHEET_FULL \
                   --instruction_type $instr \
                   $EXT_FLAG \
                   $BLOCK_ARGS \
                   $DIALECT_ARGS \
                   $SV_ARGS \
                   $LABEL_ARGS \
                   $CTX_ARGS \
                   --output_dir $OUTDIR"
            fi

          done
        done
      done
    done
  done
done
