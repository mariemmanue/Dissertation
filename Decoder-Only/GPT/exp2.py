# exp2.py - Structured syntactic analysis approach
import os
import pandas as pd
from collections import defaultdict
import csv
import json
import time
from tqdm import tqdm
import getpass
import re

from openai import OpenAI
import tiktoken
import argparse
from gpt_experiments import (
    EXTENDED_FEATURES,
    MASIS_FEATURES, 
    MASIS_FEATURE_BLOCK, 
    NEW_FEATURE_BLOCK, build_system_msg,
)
# Initialize global variables
total_input_tokens = 0
total_output_tokens = 0
api_call_count = 0

OPENAI_MODEL_NAME = "gpt-5"
# tokenizer for logging
enc = tiktoken.encoding_for_model("gpt-5")

# Define constants for features


# Define feature blocks containing the rules for each feature
# Same feature definitions as befor
# Utility function definitions
def drop_features_column(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=['FEATURES', 'Source'], errors='ignore')



def build_feature_instruction(feature_name, feature_rule):
    """Build structured instruction for a single feature"""
    return f"""
FEATURE: {feature_name}
Decision rule: {feature_rule}

Task: Determine whether {feature_name} is PRESENT (1) or NOT PRESENT (0).

Step 1 – Identify the main clause and quote it exactly.
Step 2 – Identify relevant syntactic components (subject, verb, auxiliaries, etc.) as needed for this feature.
Step 3 – Apply the grammatical rule above.
Step 4 – Provide a binary decision with metalinguistic rationale.

Respond ONLY in this format:

MAIN CLAUSE: "<quoted clause>"
SYNTACTIC ANALYSIS: "<identify subject, verb, and other relevant components>"
RATIONALE: "<1 sentence stating whether rule applies or not, referencing the grammatical rule and specific syntactic structures>"
ANSWER: <1 or 0>
"""


def build_system_response_instructions(utterance, features, feature_rules_dict):
    """Build instructions for structured response format"""
    feature_instructions = []
    for feat in features:
        rule = feature_rules_dict.get(feat, "")
        feature_instructions.append(build_feature_instruction(feat, rule))
    
    feature_instructions_text = "\n".join(feature_instructions)
    
    return (
        "Now analyze this utterance:\n"
        f"UTTERANCE: {utterance}\n\n"
        "For EACH feature below, provide your analysis using the structured format.\n"
        "You must analyze ALL features, even if most are 0.\n\n"
        f"{feature_instructions_text}\n\n"
        "Return ONLY a single JSON object with EXACTLY these keys, in this order:\n"
        f"[{', '.join(f'\"{f}\"' for f in features)}]\n\n"
        "For each key, set:\n"
        "- \"main_clause\": \"<quoted main clause>\"\n"
        "- \"syntactic_analysis\": \"<subject, verb, and relevant components>\"\n"
        "- \"rationale\": \"<1 sentence metalinguistic justification>\"\n"
        "- \"value\": 1 or 0\n\n"
        "Example output format (structure only):\n"
        "{\n"
        '  "zero-poss": {\n'
        '    "main_clause": "That they dad car.",\n'
        '    "syntactic_analysis": "Subject: \'that\'; Possessive construction: noun-noun juxtaposition without possessive morpheme",\n'
        '    "rationale": "The construction uses noun-noun juxtaposition (dad car) without possessive morpheme, therefore zero-poss is present (1).",\n'
        '    "value": 1\n'
        '  },\n'
        '  "zero-copula": {\n'
        '    "main_clause": "That they dad car.",\n'
        '    "syntactic_analysis": "Subject: \'that\'; Predicate: \'they dad car\'; Missing copula between subject and predicate",\n'
        '    "rationale": "The main clause lacks required copula between subject and predicate, therefore zero-copula is present (1).",\n'
        '    "value": 1\n'
        '  },\n'
        "  ...\n"
        "}\n"
        "Do not add fields. Do not change key names. Do not explain outside the JSON.\n"
    )

def extract_feature_rules(feature_block):
    """Extract decision rules for each feature from the feature block"""
    rules_dict = {}
    lines = feature_block.split('\n')
    current_feature = None
    current_rule = []
    
    for line in lines:
        # Match feature number/name pattern
        if re.match(r'^\d+[a-z]?\.\s+(\S+)', line) or re.match(r'^(\S+):', line):
            # Save previous feature if exists
            if current_feature and current_rule:
                rules_dict[current_feature] = ' '.join(current_rule).strip()
            
            # Extract feature name
            match = re.match(r'^\d+[a-z]?\.\s+(\S+)', line) or re.match(r'^(\S+):', line)
            if match:
                current_feature = match.group(1)
                current_rule = []
                # Check if rule starts on same line
                rule_part = line.split(':', 1)
                if len(rule_part) > 1:
                    current_rule.append(rule_part[1].strip())
        elif line.strip().startswith('Decision rule:'):
            # Extract rule text
            rule_text = line.split('Decision rule:', 1)[1].strip()
            current_rule.append(rule_text)
        elif current_feature and (line.strip().startswith('+') or line.strip().startswith('-') or line.strip().startswith('Note')):
            # Skip examples and notes
            continue
        elif current_feature and line.strip():
            # Continue collecting rule text
            current_rule.append(line.strip())
    
    # Save last feature
    if current_feature and current_rule:
        rules_dict[current_feature] = ' '.join(current_rule).strip()
    
    return rules_dict

def _build_gpt_prompt_generic(utterance, features, feature_block):
    """Build GPT prompt with structured syntactic analysis format"""
    system_msg = build_system_msg()
    
    # Extract feature rules
    feature_rules_dict = extract_feature_rules(feature_block)
    
    response_instructions = build_system_response_instructions(utterance, features, feature_rules_dict)

    user_msg = {
        "role": "user",
        "content": feature_block + "\n\n" + response_instructions,
    }

    return [system_msg, user_msg]

def build_gpt_prompt_masis(utterance, masis_features, feature_block_17):
    return _build_gpt_prompt_generic(
        utterance=utterance,
        features=masis_features,
        feature_block=feature_block_17,
    )

def build_gpt_prompt_extended(utterance, extended_features, new_features):
    return _build_gpt_prompt_generic(
        utterance=utterance,
        features=extended_features,
        feature_block=new_features,
    )

def build_gpt_prompt_extended_context(utterance, features, feature_block, left_context=None, right_context=None):
    """Build prompt with context, but still require main clause identification"""
    system_msg = build_system_msg()
    
    left = left_context if left_context else "(none)"
    right = right_context if right_context else "(none)"

    context_block = (
        "You will see a short window of discourse:\n\n"
        f"CONTEXT BEFORE:\n{left}\n\n"
        f"TARGET UTTERANCE (the one you must tag):\n{utterance}\n\n"
        f"CONTEXT AFTER:\n{right}\n\n"
        "IMPORTANT:\n"
        "- Make ALL feature decisions ONLY about the TARGET UTTERANCE.\n"
        "- Identify the MAIN CLAUSE of the TARGET UTTERANCE first.\n"
        "- Use CONTEXT BEFORE/AFTER ONLY to clarify tense, aspect, or reference for the main clause analysis.\n"
        "- Do NOT mark a feature just because it appears in context; it must be realized in the TARGET UTTERANCE's main clause.\n"
    )

    # Extract feature rules
    feature_rules_dict = extract_feature_rules(feature_block)
    response_instructions = build_system_response_instructions(utterance, features, feature_rules_dict)

    user_msg = {
        "role": "user",
        "content": feature_block + "\n\n" + context_block + "\n\n" + response_instructions,
    }

    return [system_msg, user_msg]

# Final summary after processing all sentences
def print_final_usage_summary():
    total_tokens = total_input_tokens + total_output_tokens

    print("\n===== FINAL USAGE SUMMARY =====")
    print(f"Total API Calls: {api_call_count}")
    print(f"Total Input Tokens: {total_input_tokens}")
    print(f"Total Output Tokens: {total_output_tokens}")
    print(f"Total Tokens: {total_tokens}")
    print("===================================\n")

def parse_output_json(raw_str, features):
    """Parse JSON output, extracting values and rationales"""
    data = json.loads(raw_str)

    vals = {}
    rats = {}
    main_clauses = {}
    syntactic_analyses = {}
    
    for feat in features:
        entry = data.get(feat, {})
        vals[feat] = int(entry.get("value", 0))
        rats[feat] = str(entry.get("rationale", "")).strip()
        main_clauses[feat] = str(entry.get("main_clause", "")).strip()
        syntactic_analyses[feat] = str(entry.get("syntactic_analysis", "")).strip()
    
    return vals, rats, main_clauses, syntactic_analyses

# Query GPT functions
def query_gpt(client, enc, sentence, features, masis_features, feature_block_17, new_features, extended=False, max_retries=15, base_delay=3):
    global api_call_count, total_input_tokens, total_output_tokens

    if extended:
        messages = build_gpt_prompt_extended(sentence, features, new_features)
    else:
        messages = build_gpt_prompt_masis(sentence, masis_features, feature_block_17)

    input_tokens = sum(len(enc.encode(msg["content"])) for msg in messages)
    total_input_tokens += input_tokens

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL_NAME,
                messages=messages,
            )

            output_text = resp.choices[0].message.content
            output_tokens = len(enc.encode(output_text))
            total_output_tokens += output_tokens
            api_call_count += 1

            print(output_text)
            print(
                f"API Call #{api_call_count} | "
                f"Input Tokens: {input_tokens} | "
                f"Output Tokens: {output_tokens} | "
                f"Total: {total_input_tokens + total_output_tokens}"
            )
            return output_text

        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait_time = base_delay * (2 ** attempt)
                print(f"Rate limit hit. Retrying in {wait_time}s... (Attempt {attempt + 1})")
                time.sleep(wait_time)
            else:
                print(f"Error on sentence: {sentence[:40]}... → {e}")
                return None

    print(f"Failed after {max_retries} retries.")
    return None

def query_gpt_context(client, enc, sentence, features, feature_block, left_context=None, right_context=None, max_retries=15, base_delay=3):
    global api_call_count, total_input_tokens, total_output_tokens

    messages = build_gpt_prompt_extended_context(sentence, features, feature_block, left_context, right_context)

    input_tokens = sum(len(enc.encode(msg["content"])) for msg in messages)
    total_input_tokens += input_tokens

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL_NAME,
                messages=messages,
            )
            output_text = resp.choices[0].message.content
            output_tokens = len(enc.encode(output_text))
            total_output_tokens += output_tokens
            api_call_count += 1
            print(output_text)
            print(
                f"API Call #{api_call_count} | "
                f"Input Tokens: {input_tokens} | "
                f"Output Tokens: {output_tokens} | "
                f"Total: {total_input_tokens + total_output_tokens}"
            )
            return output_text

        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait_time = base_delay * (2 ** attempt)
                print(f"Rate limit hit. Retrying in {wait_time}s... (Attempt {attempt + 1})")
                time.sleep(wait_time)
            else:
                print(f"Error on sentence: {sentence[:40]}... → {e}")
                return None

    print(f"Failed after {max_retries} retries.")
    return None

# Main Function
def main():
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Run GPT Experiments with Structured Syntactic Analysis.")
    parser.add_argument("--file", type=str, help="Input Excel file path", required=True)
    parser.add_argument("--sheet", type=str, help="Sheet name for GPT experiment", required=True)
    parser.add_argument("--extended", action="store_true", help="Use extended feature set")
    parser.add_argument("--context", action="store_true", help="Use context in the experiments")
    parser.add_argument("--output_dir", type=str, help="Output directory for results", required=True)
    args = parser.parse_args()

    file_title = os.path.splitext(os.path.basename(args.file))[0]
    output_dir = os.path.join(args.output_dir, file_title)
    os.makedirs(output_dir, exist_ok=True)

    sheets = pd.read_excel(args.file, sheet_name=None)
    gold_df = sheets["Gold"]
    gold_df = gold_df.dropna(subset=["sentence"]).reset_index(drop=True)

    eval_sentences = gold_df["sentence"].dropna().tolist()

    print(f"Number of sentences to evaluate: {len(eval_sentences)}")

    # Read the OpenAI API key from an environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    
    client = OpenAI(api_key=openai_api_key)
    enc = tiktoken.get_encoding("p50k_base")

    results = []
    rationale_rows = []
    main_clause_rows = []
    syntactic_analysis_rows = []

    preds_path = os.path.join(output_dir, args.sheet + "_predictions.csv")
    rats_path = os.path.join(output_dir, args.sheet + "_rationales.csv")
    main_clause_path = os.path.join(output_dir, args.sheet + "_main_clauses.csv")
    syntactic_path = os.path.join(output_dir, args.sheet + "_syntactic_analyses.csv")

    preds_header = ["sentence"] + EXTENDED_FEATURES
    rats_header = ["sentence"] + [f"{feat}" for feat in EXTENDED_FEATURES]
    main_clause_header = ["sentence"] + [f"{feat}_main_clause" for feat in EXTENDED_FEATURES]
    syntactic_header = ["sentence"] + [f"{feat}_syntactic" for feat in EXTENDED_FEATURES]

    if not os.path.exists(preds_path):
        with open(preds_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(preds_header)

    if not os.path.exists(rats_path):
        with open(rats_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(rats_header)

    if not os.path.exists(main_clause_path):
        with open(main_clause_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(main_clause_header)

    if not os.path.exists(syntactic_path):
        with open(syntactic_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(syntactic_header)

    USE_EXTENDED = args.extended
    CURRENT_FEATURES = EXTENDED_FEATURES if USE_EXTENDED else MASIS_FEATURES
    FEATURE_BLOCK = NEW_FEATURE_BLOCK if USE_EXTENDED else MASIS_FEATURE_BLOCK

    # Iterating through sentences to evaluate
    for sentence in tqdm(eval_sentences, desc="Evaluating sentences"):
        if args.context:
            prev_sentence = gold_df.loc[gold_df["sentence"] == sentence, "prev_sentence"].values[0] if "prev_sentence" in gold_df.columns else None
            next_sentence = gold_df.loc[gold_df["sentence"] == sentence, "next_sentence"].values[0] if "next_sentence" in gold_df.columns else None
            raw = query_gpt_context(client, enc, sentence, CURRENT_FEATURES, FEATURE_BLOCK, left_context=prev_sentence, right_context=next_sentence)
        else:
            raw = query_gpt(client, enc, sentence, CURRENT_FEATURES, MASIS_FEATURES, MASIS_FEATURE_BLOCK, NEW_FEATURE_BLOCK, extended=USE_EXTENDED)

        if not raw:
            continue

        try:
            vals, rats, main_clauses, syntactic_analyses = parse_output_json(raw, features=CURRENT_FEATURES)
        except json.JSONDecodeError as e:
            print(f"JSON parse fail on sentence: {sentence}\nError: {e}")
            print(raw)
            continue

        pred_row = {"sentence": sentence}
        pred_row.update(vals)
        results.append(pred_row)

        rat_row = {"sentence": sentence}
        main_clause_row = {"sentence": sentence}
        syntactic_row = {"sentence": sentence}
        
        for feat in CURRENT_FEATURES:
            rat_row[f"{feat}"] = rats.get(feat, "")
            main_clause_row[f"{feat}_main_clause"] = main_clauses.get(feat, "")
            syntactic_row[f"{feat}_syntactic"] = syntactic_analyses.get(feat, "")
        
        rationale_rows.append(rat_row)
        main_clause_rows.append(main_clause_row)
        syntactic_analysis_rows.append(syntactic_row)

        with open(preds_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([sentence] + [vals.get(feat, "") for feat in CURRENT_FEATURES])

        with open(rats_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([sentence] + [rats.get(feat, "") for feat in CURRENT_FEATURES])

        with open(main_clause_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([sentence] + [main_clauses.get(feat, "") for feat in CURRENT_FEATURES])

        with open(syntactic_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([sentence] + [syntactic_analyses.get(feat, "") for feat in CURRENT_FEATURES])

    # Write the predictions and rationales directly to the Excel file
    predictions_df = pd.DataFrame(results)
    rationales_df = pd.DataFrame(rationale_rows)
    
    with pd.ExcelWriter(args.file, mode='a', if_sheet_exists='replace') as writer:
        predictions_df.to_excel(writer, sheet_name=args.sheet, index=False)

    print_final_usage_summary()

if __name__ == "__main__":
    main()
