import os

# --- USER CONFIGURATION ---
# Add more models here if needed. 
# The script uses the first matching key in MODEL_RESOURCES to set partition/memory.
MODELS_TO_RUN = ["microsoft/phi-4"]

# Resource mapping
MODEL_RESOURCES = {
    "microsoft/phi-4": {"partition": "jag",      "memory": "40G", "cores": 2},
    "google/gemma-7b": {"partition": "standard", "memory": "32G", "cores": 4},
    "default":         {"partition": "standard", "memory": "32G", "cores": 2}
}

# Experiment Factors (2 x 2 x 2 x 4 = 32 configs)
SHOT_TYPES = ["ZS", "ZSCOT", "ICL", "FSCOT"]
CONTEXTS = ["noCTX", "CTX"]
LEGITIMACY = ["nolegit", "legit"]
OUTPUTS = ["labels", "rats"]

def get_resources(model_name):
    """Finds the correct resources for a given model name."""
    for key, res in MODEL_RESOURCES.items():
        if key in model_name:
            return res
    return MODEL_RESOURCES["default"]

def generate_bash_script(model_name):
    res = get_resources(model_name)
    # create a clean filename for the shell script
    safe_name = model_name.split('/')[-1]
    script_name = f"submit_all_{safe_name}.sh"
    
    with open(script_name, "w") as f:
        # Script Header
        f.write("#!/bin/bash\n")
        f.write(f"# Auto-generated submitter for {model_name}\n")
        f.write(f"# Resources: {res['partition']} | {res['memory']} | {res['cores']} cores\n")
        f.write("mkdir -p slurm_logs\n\n")
        
        job_count = 0
        
        for shot in SHOT_TYPES:
            for ctx in CONTEXTS:
                for leg in LEGITIMACY:
                    for out in OUTPUTS:
                        
                        # 1. Create Unique Config ID
                        config_id = f"{shot}_{ctx}_{leg}_{out}"
                        
                        # 2. Build Python Command
                        py_cmd = (
                            f"python Decoder-Only/multi_prompt_configs.py "
                            f"--model_name {model_name} "
                            f"--output_dir Decoder-Only/Results/{config_id}"
                        )
                        
                        # Flags based on factors
                        if shot == "ZS": py_cmd += " --prompt_type zero-shot"
                        elif shot == "ZSCOT": py_cmd += " --prompt_type zero-shot-cot"
                        elif shot == "ICL": py_cmd += " --prompt_type few-shot"
                        elif shot == "FSCOT": py_cmd += " --prompt_type few-shot-cot"
                        
                        if ctx == "CTX": py_cmd += " --use_context"
                        if leg == "legit": py_cmd += " --use_legitimacy"
                        
                        if out == "rats": py_cmd += " --output_format rationales"
                        else: py_cmd += " --output_format labels"

                        # 3. Build SLURM Command
                        # We use 'nlprun' to submit. No '&' needed; it submits and returns.
                        slurm_cmd = (
                            f'echo "Submitting {config_id}..."\n'
                            f'nlprun -q {res["partition"]} -p standard -r {res["memory"]} -c {res["cores"]} '
                            f'-n {config_id} '
                            f'-o slurm_logs/{config_id}-%j.out '
                            f'"cd /nlp/scr/mtano/Dissertation && '
                            f'. /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && '
                            f'conda activate cgedit && '
                            f'{py_cmd}"\n\n'
                        )
                        
                        f.write(slurm_cmd)
                        job_count += 1
        
        f.write(f"echo 'Done. Submitted {job_count} jobs to the scheduler.'\n")
    
    print(f"Generated: {script_name} ({job_count} jobs)")

if __name__ == "__main__":
    for model in MODELS_TO_RUN:
        generate_bash_script(model)
