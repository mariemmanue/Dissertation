#!/usr/bin/env python3
"""
Generates a shell script with all experimental configurations as nlprun jobs.

Grid per model:
  instruction_type:    zero_shot, few_shot, zero_shot_cot, few_shot_cot  (4)
  context:             off, on+single_turn, on+two_turn                  (3)
  dialect_legitimacy:  off, on                                           (2)
  ────────────────────────────────────────────────────────────────────────
  Total per model:     4 × 3 × 2 = 24

Models: GPT-5.2 Instant, GPT-5.2 Thinking (med/high), Gemini 2.5 Flash,
        Phi-4, Phi-4-reasoning, Qwen 2.5-7B, Qwen3-32B (think/no-think),
        QwQ-32B, Llama 3.1-70B
Total jobs: 288
"""

import itertools

# ==================== CONFIG ====================

INPUT_FILE = "Datasets/FullTest_Final.xlsx"
GOLD_FILE = "Datasets/FullTest_Final.xlsx"  # gold labels source (CSV or Excel with Gold sheet)
OUTPUT_FORMAT = "markdown"  # all models use markdown
CONDA_ENV = "cgedit"
BASE_DIR = "/nlp/scr/mtano/Dissertation"
CONDA_INIT = ". /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh"

MODELS = {
    # ── Existing models ──
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
    "qwen25_7b": {
        "backend": "qwen",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "output_dir": "Decoder-Only/Qwen2.5/data",
        "log_dir": "Decoder-Only/Qwen2.5/slurm_logs",
        "nlprun_flags": "-g 1 -q sphinx -p standard -r 100G -c 4",
    },
    # ── New: Gemini 3 Pro (reasoning, thinking=high) ──
    "gemini3_pro": {
        "backend": "gemini3",
        "model": "gemini-3-pro-preview",
        "output_dir": "Decoder-Only/Gemini3_Pro/data",
        "log_dir": "Decoder-Only/Gemini3_Pro/slurm_logs",
        "nlprun_flags": "-q jag -p standard -r 40G -c 2",
        "extra_args": "--thinking_level high",
    },
    # ── New: GPT-4.1 (non-reasoning, still available via API) ──
    "gpt41": {
        "backend": "openai",
        "model": "gpt-4.1",
        "output_dir": "Decoder-Only/GPT41/data",
        "log_dir": "Decoder-Only/GPT41/slurm_logs",
        "nlprun_flags": "-q jag -p standard -r 40G -c 2",
    },
    # ── New: GPT-5.2 family ──
    "gpt52_instant": {
        "backend": "openai",
        "model": "gpt-5.2-chat-latest",
        "output_dir": "Decoder-Only/GPT52_Instant/data",
        "log_dir": "Decoder-Only/GPT52_Instant/slurm_logs",
        "nlprun_flags": "-q jag -p standard -r 40G -c 2",
    },
    "gpt52_think_med": {
        "backend": "openai_reasoning",
        "model": "gpt-5.2",
        "output_dir": "Decoder-Only/GPT52_Thinking_Med/data",
        "log_dir": "Decoder-Only/GPT52_Thinking_Med/slurm_logs",
        "nlprun_flags": "-q jag -p standard -r 40G -c 2",
        "extra_args": "--reasoning_effort medium",
    },
    "gpt52_think_high": {
        "backend": "openai_reasoning",
        "model": "gpt-5.2",
        "output_dir": "Decoder-Only/GPT52_Thinking_High/data",
        "log_dir": "Decoder-Only/GPT52_Thinking_High/slurm_logs",
        "nlprun_flags": "-q jag -p standard -r 40G -c 2",
        "extra_args": "--reasoning_effort high",
    },
    # ── New: Phi-4-reasoning (pairs with phi4) ──
    "phi4_reasoning": {
        "backend": "phi_reasoning",
        "model": "microsoft/Phi-4-reasoning",
        "output_dir": "Decoder-Only/Phi-4-reasoning/data",
        "log_dir": "Decoder-Only/Phi-4-reasoning/slurm_logs",
        "nlprun_flags": "-g 1 -q sphinx -p standard -r 100G -c 4",
    },
    # ── New: Llama 3.1-70B (large open-source baseline) ──
    "llama70b": {
        "backend": "llama",
        "model": "meta-llama/Llama-3.1-70B-Instruct",
        "output_dir": "Decoder-Only/Llama-3.1-70B/data",
        "log_dir": "Decoder-Only/Llama-3.1-70B/slurm_logs",
        "nlprun_flags": "-g 4 -q sphinx -p standard -r 300G -c 8",
    },
    # ── New: Qwen3-32B non-thinking (non-reasoning baseline) ──
    "qwen3_32b": {
        "backend": "qwen3",
        "model": "Qwen/Qwen3-32B",
        "output_dir": "Decoder-Only/Qwen3-32B/data",
        "log_dir": "Decoder-Only/Qwen3-32B/slurm_logs",
        "nlprun_flags": "-g 2 -q sphinx -p standard -r 200G -c 4",
    },
    # ── New: Qwen3-32B thinking (reasoning mode) ──
    "qwen3_32b_think": {
        "backend": "qwen3_thinking",
        "model": "Qwen/Qwen3-32B",
        "output_dir": "Decoder-Only/Qwen3-32B-Thinking/data",
        "log_dir": "Decoder-Only/Qwen3-32B-Thinking/slurm_logs",
        "nlprun_flags": "-g 2 -q sphinx -p standard -r 200G -c 4",
    },
    # ── New: QwQ-32B (always-reasoning) ──
    "qwq_32b": {
        "backend": "qwq",
        "model": "Qwen/QwQ-32B",
        "output_dir": "Decoder-Only/QwQ-32B/data",
        "log_dir": "Decoder-Only/QwQ-32B/slurm_logs",
        "nlprun_flags": "-g 2 -q sphinx -p standard -r 200G -c 4",
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

def generate_command(model_key, inst_type, ctx_setting, dialect_leg, job_num=0):
    m = MODELS[model_key]
    sheet = make_sheet_name(model_key, inst_type, ctx_setting, dialect_leg)
    job = f"{job_num:02d}_{make_job_name(model_key, inst_type, ctx_setting, dialect_leg)}"

    # Build python args
    py_args = [
        f"--file {INPUT_FILE}",
        f"--gold {GOLD_FILE}",
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

    # Append any model-specific extra args (e.g. --reasoning_effort)
    if "extra_args" in m:
        py_args.append(m["extra_args"])

    py_args_str = " \\\n    ".join(py_args)

    cmd = (
        f'nlprun {m["nlprun_flags"]} \\\n'
        f'  -n {job} \\\n'
        f'  -o {m["log_dir"]}/%x.out \\\n'
        f'  "cd {BASE_DIR} && \\\n'
        f'   {CONDA_INIT} && \\\n'
        f'   conda activate {CONDA_ENV} && \\\n'
        f'   python Decoder-Only/multi_prompt_configs.py \\\n'
        f'    {py_args_str}"'
    )
    return cmd, sheet, job


def main():
    model_keys = list(MODELS.keys())
    total_jobs = len(MODELS) * len(INSTRUCTION_TYPES) * len(CONTEXT_SETTINGS) * len(DIALECT_LEGITIMACY)

    lines = [
        "#!/bin/bash",
        "# Auto-generated experimental run script",
        f"# Total jobs: {total_jobs}",
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
        f"#   ./run_all_experiments.sh              # launch all {total_jobs} jobs",
        "#   ./run_all_experiments.sh phi4           # launch only phi4 jobs (24)",
        "#   ./run_all_experiments.sh gemini         # launch only gemini jobs (24)",
        "#   ./run_all_experiments.sh gpt52_instant  # launch only GPT-5.2 Instant jobs (24)",
        "#",
        f"# Available model keys: {', '.join(model_keys)}",
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
        lines.append(f'mkdir -p {MODELS[model_key]["log_dir"]}')
        lines.append(f'mkdir -p {MODELS[model_key]["output_dir"]}')
        lines.append("")

        model_job_num = 0
        for inst_type, ctx_setting, dialect_leg in itertools.product(
            INSTRUCTION_TYPES, CONTEXT_SETTINGS, DIALECT_LEGITIMACY
        ):
            model_job_num += 1
            cmd, sheet, job = generate_command(model_key, inst_type, ctx_setting, dialect_leg, job_num=model_job_num)
            job_count += 1

            lines.append(f"echo \"[{job_count:03d}] Launching: {sheet}\"")
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
