#!/usr/bin/env python3
"""
Generates a shell script with all experimental configurations as nlprun jobs.

Grid per model:
  instruction_type:    zero_shot, few_shot, zero_shot_cot, few_shot_cot  (4)
  context:             off, on+single_turn, on+two_turn                  (3)
  dialect_legitimacy:  off, on                                           (2)
  ────────────────────────────────────────────────────────────────────────
  Total per model:     4 × 3 × 2 = 24

Models: GPT-4o, Gemini 2.5 Flash, Phi-4
Total jobs: 72
"""

import itertools

# ==================== CONFIG ====================

INPUT_FILE = "Datasets/FullTest_Final.xlsx"
OUTPUT_FORMAT = "markdown"  # all models use markdown
CONDA_ENV = "cgedit"
BASE_DIR = "/nlp/scr/mtano/Dissertation"
CONDA_INIT = ". /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh"

MODELS = {
    "phi4": {
        "backend": "phi",
        "model": "microsoft/phi-4",
        "output_dir": "Decoder-Only/Phi-4/data",
        "log_dir": "Decoder-Only/Phi-4/slurm_logs",
        "nlprun_flags": "-g 1 -q sphinx -p standard -r 100G -c 4",
    },
    "gemini": {
        "backend": "gemini",
        "model": "gemini-2.5-flash",
        "output_dir": "Decoder-Only/Gemini/data",
        "log_dir": "Decoder-Only/Gemini/slurm_logs",
        "nlprun_flags": "-q jag -p standard -r 40G -c 2",
    },
    "gpt4o": {
        "backend": "openai",
        "model": "gpt-4o",
        "output_dir": "Decoder-Only/GPT4o/data",
        "log_dir": "Decoder-Only/GPT4o/slurm_logs",
        "nlprun_flags": "-q jag -p standard -r 40G -c 2",
    },
}

# Experimental conditions
INSTRUCTION_TYPES = ["zero_shot", "few_shot", "zero_shot_cot", "few_shot_cot"]
CONTEXT_SETTINGS = [
    {"context": False, "context_mode": None},
    {"context": True,  "context_mode": "single_turn"},
    {"context": True,  "context_mode": "two_turn"},
]
DIALECT_LEGITIMACY = [False, True]

# ==================== NAMING ====================

INST_SHORT = {
    "zero_shot": "ZS",
    "few_shot": "FS",
    "zero_shot_cot": "ZScot",
    "few_shot_cot": "FScot",
}

def make_sheet_name(model_key, inst_type, ctx_setting, dialect_leg):
    """Generate a unique, readable sheet name for this configuration."""
    parts = [
        model_key.upper(),
        INST_SHORT[inst_type],
    ]

    if not ctx_setting["context"]:
        parts.append("noCTX")
    elif ctx_setting["context_mode"] == "single_turn":
        parts.append("CTX1t")
    else:
        parts.append("CTX2t")

    parts.append("Leg" if dialect_leg else "noLeg")

    return "_".join(parts)


def make_job_name(model_key, inst_type, ctx_setting, dialect_leg):
    """Short job name for slurm."""
    return make_sheet_name(model_key, inst_type, ctx_setting, dialect_leg).lower()


# ==================== GENERATE ====================

def generate_command(model_key, inst_type, ctx_setting, dialect_leg):
    m = MODELS[model_key]
    sheet = make_sheet_name(model_key, inst_type, ctx_setting, dialect_leg)
    job = make_job_name(model_key, inst_type, ctx_setting, dialect_leg)

    # Build python args
    py_args = [
        f"--file {INPUT_FILE}",
        f"--model {m['model']}",
        f"--backend {m['backend']}",
        f"--sheet {sheet}",
        f"--instruction_type {inst_type}",
        "--extended",
        f"--output_format {OUTPUT_FORMAT}",
        f"--output_dir {m['output_dir']}",
        "--dump_prompt",
    ]

    if ctx_setting["context"]:
        py_args.append("--context")
        py_args.append(f"--context_mode {ctx_setting['context_mode']}")

    if dialect_leg:
        py_args.append("--dialect_legitimacy")

    py_args_str = " \\\n    ".join(py_args)

    cmd = (
        f'nlprun {m["nlprun_flags"]} \\\n'
        f'  -n {job} \\\n'
        f'  -o {m["log_dir"]}/%x-%j.out \\\n'
        f'  "cd {BASE_DIR} && \\\n'
        f'   {CONDA_INIT} && \\\n'
        f'   conda activate {CONDA_ENV} && \\\n'
        f'   python Decoder-Only/multi_prompt_configs.py \\\n'
        f'    {py_args_str}"'
    )
    return cmd, sheet, job


def main():
    lines = [
        "#!/bin/bash",
        "# Auto-generated experimental run script",
        f"# Total jobs: {len(MODELS) * len(INSTRUCTION_TYPES) * len(CONTEXT_SETTINGS) * len(DIALECT_LEGITIMACY)}",
        "#",
        "# Grid per model:",
        "#   instruction_type:   zero_shot, few_shot, zero_shot_cot, few_shot_cot  (4)",
        "#   context:            off, single_turn, two_turn                        (3)",
        "#   dialect_legitimacy: off, on                                           (2)",
        "#   output_format:      markdown (all models)                             (1)",
        "#   ──────────────────────────────────────────────────────────────────────",
        "#   Total per model:    24",
        "#",
        "# Usage:",
        "#   chmod +x run_all_experiments.sh",
        "#   ./run_all_experiments.sh           # launch all 72 jobs",
        "#   ./run_all_experiments.sh phi4      # launch only phi4 jobs (24)",
        "#   ./run_all_experiments.sh gemini    # launch only gemini jobs (24)",
        "#   ./run_all_experiments.sh gpt4o     # launch only gpt4o jobs (24)",
        "",
        "set -e",
        "",
        'MODEL_FILTER="${1:-all}"',
        "",
    ]

    job_count = 0
    for model_key in MODELS:
        lines.append(f"# {'='*60}")
        lines.append(f"# {model_key.upper()} — {MODELS[model_key]['model']}")
        lines.append(f"# {'='*60}")
        lines.append("")
        lines.append(f'if [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "{model_key}" ]]; then')
        lines.append("")

        for inst_type, ctx_setting, dialect_leg in itertools.product(
            INSTRUCTION_TYPES, CONTEXT_SETTINGS, DIALECT_LEGITIMACY
        ):
            cmd, sheet, job = generate_command(model_key, inst_type, ctx_setting, dialect_leg)
            job_count += 1

            lines.append(f"echo \"[{job_count:02d}] Launching: {sheet}\"")
            lines.append(cmd)
            lines.append("")

        lines.append("fi")
        lines.append("")

    lines.append(f'echo "Done. Submitted jobs for $MODEL_FILTER."')

    script = "\n".join(lines) + "\n"

    out_path = "Decoder-Only/run_all_experiments.sh"
    with open(out_path, "w") as f:
        f.write(script)

    print(f"Generated {out_path}")
    print(f"Total jobs: {job_count}")
    print(f"\nTo run:  chmod +x {out_path} && ./{out_path}")
    print(f"Or:      ./{out_path} phi4     # just one model")


if __name__ == "__main__":
    main()
