# gpt_experiments.py
import os
import pandas as pd
from collections import defaultdict
import csv
import json
import time
from tqdm import tqdm
import getpass

from openai import OpenAI
import tiktoken
import argparse

# Initialize global variables
total_input_tokens = 0
total_output_tokens = 0
api_call_count = 0

# Ensure paths

"""nlprun -q jag -p standard -r 40G -c 2 \
  -n gpt-exp-aae \
  -o slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation/Decoder-Only/GPT && \
   mkdir -p slurm_logs data/results && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   export HF_HOME=/nlp/scr/mtano/hf_home && \
   python gpt_experiments.py \
     --file data/Run1.xlsx \
     --sheet GPT-Exp2 \
     --extended
      --output_dir data/results" """

"""nlprun -q jag -p standard -r 40G -c 2 \
  -n gpt-exp-aae \
  -o slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation/Decoder-Only/GPT && \
   mkdir -p slurm_logs data/results && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   export HF_HOME=/nlp/scr/mtano/hf_home && \
   python gpt_experiments.py \
     --file data/Run1.xlsx \
     --sheet GPT-Exp1
      --output_dir data/results" """

OPENAI_MODEL_NAME = "gpt-5"
# tokenizer for logging
enc = tiktoken.encoding_for_model("gpt-5")

# Define constants for features
MASIS_FEATURES = [
    "zero-poss", "zero-copula", "double-tense", "be-construction", "resultant-done", "finna", "come", "double-modal", 
    "multiple-neg", "neg-inversion", "n-inv-neg-concord", "aint", "zero-3sg-pres-s", "is-was-gen", "zero-pl-s", 
    "double-object", "wh-qu1", "wh-qu2"
]

EXTENDED_FEATURES = [ 
    "zero-poss", "zero-copula", "double-tense", "be-construction", "resultant-done", "finna", "come", "double-modal", 
    "multiple-neg", "neg-inversion", "n-inv-neg-concord", "aint", "zero-3sg-pres-s", "is-was-gen", "zero-pl-s", 
    "double-object", "wh-qu1", "wh-qu2", "existential-it", "demonstrative-them", "appositive-pleonastic-pronoun", 
    "bin", "verb-stem", "past-tense-swap", "zero-rel-pronoun", "preterite-had"
]

# Define feature blocks containing the rules for each feature
# rule-first definitions with 1 pos + 1 near-miss for hard ones
MASIS_FEATURE_BLOCK = """
1. zero-poss
   Decision rule: Possession is expressed without a possessive morpheme ('s) — either through noun–noun juxtaposition (dad car) or by using bare or nonstandard possessive pronouns (they car).
   + Example (1): "That they dad car." → 1
   - Near-miss (0): "That's their dad's car." → 0 (SAE possessive)

2. zero-copula
  Decision rule: Required is/are/was/were is missing either (a) between the subject and predicate, or (b) as an auxiliary before a V-ing verb (especially in wh- or polar questions) in utterances that form a single prosodic or intonational unit (not clearly subordinate).
   + Example (1): "She real strict." → 1
   - Near-miss (0): "No problem." → 0 (ellipsis, not copula deletion)

3. double-tense
   Decision rule: Past-tense verb shows duplicated -ed marking of the past morpheme/suffix.
   + Example (1): "She likeded me the best." → 1
   - Near-miss (0): "She liked me the best." → 0

4. be-construction
   Decision rule: Uninflected be marks habitual or generalized action, and may also appear in fixed, expressive, or imminent contexts. Just exclude finite forms (is/are/was/were) and auxiliary-only uses.
   + Example (1): "They be playing outside." → 1
   - Near-miss (0): "They is playing outside." → 0 (agreement error but not habitual)

5. resultant-done
   Decision rule: Preverbal done marks completed aspect when it directly precedes a verb phrase (adverbs or discourse markers may intervene). The following verb can be past-marked or bare (esp. irregulars) as long as the completed-event reading is licensed.
   + Example (1): "He done already ate it." → 1
   - Near-miss (0): "They done it yesterday." → 0 (no past verb after 'done')

6. finna
   Decision rule: Verb 'finna' (or 'finta', 'fitna') indicates imminent/future action. Mentally expands as “fixing to / about to” when testing clause structure (e.g., He [is] finna leave).
   + Example (1): "We finna eat." → 1
   - Near-miss (0): "We gonna eat." → 0 (informal SAE but not AAE 'finna')

7. come
   Decision rule: 'come' appears preverbal and introduces a VP or V-ing form expressing speech/attitude/behavior (e.g., come tryna, come telling me).
   + Example (1): "He come talking mess again." → 1
   - Near-miss (0): "He came and talked." → 0 (motion verb, not AAE stance marker)

8. double-modal
   Decision rule: Two modals in a row (e.g. 'might could', 'used to could').
   + Example (1): "I might could go." → 1
   - Near-miss (0): "I might go." → 0

9. multiple-neg
    Decision rule: Two or more negative forms (e.g., auxiliary verbs, pronouns, or adverbs) occur within the same clause or phrase expressing a single negative meaning.
   + Example (1): "I ain't never heard of that." → 1
   - Near-miss (0): "I ain't ever heard of that." → 0 (single negation)
   Note: Multiple negation is the broad category encompassing both negative inversion (10) and negative concord (11).
   If either 10 or 11 applies, this feature (multiple-neg) must also be marked as 1.

10. neg-inversion
    Decision rule: Negative auxiliary or marker appears before the subject (a subtype of multiple negation).
    + Example (1): Don’t nobody like how they actin’. → 1
    – Near-miss (0): Nobody don’t like how they actin’. → 0 (not inversion)
    Note: Only select in addition to multiple-neg when the subject follows the negative auxiliary.

11. n-inv-neg-concord
    Decision rule: Both the subject and verb carry negative marking without inversion (a subtype of multiple negation).
    + Example (1): Nobody don’t wanna see that. → 1
    – Near-miss (0): Nobody wanna see that. → 0 (no verbal negation)
    Note: Only select in addition to multiple-neg when subject and verb are both negative but retain normal word order.

12. aint
    Decision rule: 'ain't' used as general negator for BE, HAVE, or DO.
    + Example (1): "She ain't here." → 1
    - Near-miss (0): "She isn't here." → 0

13. zero-3sg-pres-s
    Decision rule: When a 3rd person singular subject (he, she, it, nobody, somebody, etc.) appears with a bare verb or uninflected auxiliary (do → don’t, have → have, walk → walk) where SAE requires -s or does/has. Exclude non-agreeing 'is/was' forms.
    + Example (1): "She walk to they house." → 1
    - Near-miss (0): "They walk to their house." → 0 (plural subject → fine)

14. is-was-gen
    Decision rule: 'is' or 'was' occurs as a generalized or non-agreeing form—for example, used with plural or non–third person subjects. Do not mark existential/dummy 'it' constructions (It was a fight, It’s people out here), which use is/was grammatically.
    + Example (1): "They was there." → 1
    - Near-miss (0): "He was there." → 0 (SAE grammatical)

15. zero-pl-s
    Decision rule: Plural noun lacks -s but plural meaning is clear from determiner/quantifier.
    + Example (1): "She got them dog." → 1
    - Near-miss (0): "A dogs." → 0 (article–noun mismatch, not AAE plural)

16. double-object
    Decision rule: Verb takes two noun phrase objects (recipient + theme) with no preposition.
    + Example (1): "He gave him a lick." → 1
    - Near-miss (0): "He gave it to her." → 0 (preposition marks the recipient)
    Note: Exclude clausal or wh-word complements (e.g., "tell you what", "show you how").

17a. wh-qu1  (WH + copula / DO deletion)
    Decision rule: A WH-question or WH-clause where Standard English requires a form of BE or DO, but the auxiliary is missing. 
    This includes:
      • Missing copula before a predicate or locative (Where she Ø at?)
      • Missing DO-support in WH-questions (What you Ø want?, Where you Ø go?)
    + Example (1): "Where she at?" → 1  (missing 'is')
    + Example (2): "What you want?" → 1  (missing 'do')
    - Near-miss (0): "Where is she?" → 0  (auxiliary present)
    - Near-miss (0): "What did you want?" → 0  (DO-support present)
    Notes:
    • Only mark wh-qu1 when the missing auxiliary is required for a WH-question in Standard English.
    • Do not mark for simple topicalization or fragments that are not clearly questions.

17b. wh-qu2  (WH + non-standard inversion)
    Decision rule: A WH-question or WH-clause whose subject–auxiliary order departs from Standard English:
      • No inversion where SAE requires it (Where he is going? instead of Where is he going?)
      • Inversion inside embedded WH-clauses where SAE keeps declarative order 
        (I asked him could he find her instead of I asked him if he could find her).
    + Example (1): "Where he is going?" → 1  (auxiliary follows subject in a main question)
    + Example (2): "I asked him could he find her." → 1  (aux inverted in an embedded clause)
    - Near-miss (0): "Where is he going?" → 0  (standard WH inversion)
    - Near-miss (0): "I asked him if he could find her." → 0  (no embedded inversion)
    Notes:
    • Only mark wh-qu2 when word order in the WH environment is non-standard; 
      do not mark questions that are well-formed in SAE.
    • If both wh-qu1 (aux deletion) and wh-qu2 (non-inversion) clearly apply, you may mark both.
"""

NEW_FEATURE_BLOCK = MASIS_FEATURE_BLOCK + """

18. existential-it
    Decision rule: 'it' used as a dummy or existential subject where SAE would use 'there' (e.g., "It’s a dog in the yard").
    + Example (1): "It’s people out here don’t care." → 1
    - Near-miss (0): "There are people out here." → 0

19. demonstrative-them
    Decision rule: 'them' used as a demonstrative determiner instead of 'those' before a noun.
    + Example (1): "Them shoes tight." → 1
    - Near-miss (0): "Those shoes tight." → 0

20. appositive-pleonastic-pronoun
    Decision rule: Redundant or resumptive pronoun repeats the subject or object for emphasis or clarity. This includes utterances with disfluencies, interruptions, or filler words.
    + Example (1): "My dad, he told me it." → 1
    - Near-miss (0): "My mama told me that." → 0 (no pronoun repetition)

21. bin
    Decision rule: BIN/BEEN, without the auxiliary have, indicates that a state or action has been true for a long time.
    + Example (1): "She BIN married." → 1
    - Near-miss (0): "She been married for two years." → 0 (unstressed, recent past)

22. verb-stem
    Decision rule: Past meaning expressed using bare verb form instead of past-tense -ed or irregular form.
    + Example (1): "Yesterday he done walk to school." → 1 (also resultant-done → 1)
    - Near-miss (0): "He walk to school every day." → 0 (present tense, not past → zero-3sg-pres-s)

23. past-tense-swap
    Decision rule: A past participle form is used as a simple past, or a regularized past is used where an irregular form is expected.
    + Example (1): I seen him yesterday. → 1 (past participle seen used as preterite)
    – Near-miss (0): I saw him yesterday. → 0 (standard preterite form)

24. zero-rel-pronoun
    Decision rule: A relative clause modifies a noun but has no overt relative pronoun (“who/that/which”) in subject position: NP [Ø + finite clause].
    + Example (1): “There are many mothers [Ø don’t know their children].” → 1
    - Near-miss (0): “There are many mothers who don’t know their children.” → 0 (overt “who”)
    Notes: Exclude reduced relatives (“the guy wearing red”), appositives, or complement clauses not modifying the noun (“I know [that he left]”).

25. preterite-had
    Decision rule: had + past verb used to express simple past (past event with no “past-before-past” meaning). Accept overregularized forms.
    + Example (1): “The alarm next door had went off a few minutes ago.” → 1 (simple past meaning)
    - Near-miss (0): “They had seen the movie before we arrived.” → 0 (true pluperfect = past-before-past)
    Notes: Accept overregularized forms (e.g., had went, had ran). Mark 0 when “had” clearly marks pluperfect meaning (an earlier past relative to another past event).
""" 



# Utility function definitions
def drop_features_column(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=['FEATURES', 'Source'], errors='ignore')

def build_system_msg():
    return {
        "role": "system",
        "content": (
            "You are a linguist specializing in African American English (AAE). "
            "Your job is to decide, for every sentence, whether EACH AAE feature from the list "
            "is present (1) or absent (0), and to give a short grammatical rationale.\n\n"
            "GENERAL RULES:\n"
            "1. Always output ALL features, even if most are 0.\n"
            "2. Use the predefined decision rule first before looking at examples for context.\n"
            "3. If the sentence is a fragment or has disfluencies, only mark a feature if the target construction is explicitly realized (clear and not implied).\n"
            "4. If it’s ambiguous, prefer precision over recall: output 0 and explain the ambiguity.\n"
            "5. Use explicit temporal expressions (e.g., ‘yesterday’, ‘last week’) and context from surrounding verbs to clarify tense. If nearby verbs are also AAE features (e.g., verb-stem), compare their usage to SAE to avoid confusion.\n"
            "6. Examine the broader context of the utterance, focusing on key segments and ignoring fillers. Pay attention to verb construction.\n"
            "7. AAE features can co-occur (e.g., wh-question + zero copula). Mark all relevant features.\n"
            "8. Focus on verb forms and tense markers to distinguish between tense-related features like is-was-gen, zero-3sg-pres-s, verb-stem, past-tense-swap, and preterite-had. Evaluate surrounding context to avoid misclassification.\n"
            "9. Use context to identify dropped or implied subjects, unless ambiguity prevents clear identification.\n"
        ),
    }

def build_system_response_instructions(utterance, features):
    feature_list_str = ", ".join(f'"{f}"' for f in features)

    return (
        "Now analyze this utterance:\n"
        f"UTTERANCE: {utterance}\n\n"
        f"Return ONLY a single JSON object with EXACTLY these {len(features)} keys, in this order:\n"
        f"[{feature_list_str}]\n\n"
        "For each key, set:\n"
        "- \"value\": 1 or 0\n"
        "- \"rationale\": 1–2 sentences explaining the decision using the rule above.\n"
        "Example output format (structure only):\n"
        "{\n"
        '  \"zero-poss\": {\"value\": 0, \"rationale\": \"...\"},\n'
        '  \"zero-copula\": {\"value\": 1, \"rationale\": \"...\"},\n'
        "  ...\n"
        f'  \"{features[-1]}\": {{\"value\": 0, \"rationale\": \"...\"}}\n'
        "}\n"
        "Do not add fields. Do not change key names. Do not explain outside the JSON.\n"
    )

def _build_gpt_prompt_generic(utterance, features, feature_block):
    system_msg = build_system_msg()
    response_instructions = build_system_response_instructions(utterance, features)

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
        "- Use CONTEXT BEFORE/AFTER ONLY to clarify tense, aspect, or reference.\n"
        "- Do NOT mark a feature just because it appears in context; it must be realized in the TARGET UTTERANCE itself.\n"
    )

    response_instructions = build_system_response_instructions(utterance, features)

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
    data = json.loads(raw_str)

    vals = {}
    rats = {}
    for feat in features:
        entry = data.get(feat, {})
        vals[feat] = int(entry.get("value", 0))
        rats[feat] = str(entry.get("rationale", "")).strip()
    return vals, rats

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
    parser = argparse.ArgumentParser(description="Run GPT Experiments.")
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

    preds_path = os.path.join(output_dir, args.sheet + "_predictions.csv")
    rats_path = os.path.join(output_dir, args.sheet + "_rationales.csv")

    preds_header = ["sentence"] + EXTENDED_FEATURES
    rats_header = ["sentence"] + [f"{feat}" for feat in EXTENDED_FEATURES]

    if not os.path.exists(preds_path):
        with open(preds_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(preds_header)

    if not os.path.exists(rats_path):
        with open(rats_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(rats_header)

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
            vals, rats = parse_output_json(raw, features=CURRENT_FEATURES)
        except json.JSONDecodeError as e:
            print(f"JSON parse fail on sentence: {sentence}\nError: {e}")
            print(raw)
            continue

        pred_row = {"sentence": sentence}
        pred_row.update(vals)
        results.append(pred_row)

        rat_row = {"sentence": sentence}
        for feat in CURRENT_FEATURES:
            rat_row[f"{feat}"] = rats.get(feat, "")
        rationale_rows.append(rat_row)

        with open(preds_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([sentence] + [vals.get(feat, "") for feat in CURRENT_FEATURES])

        with open(rats_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([sentence] + [rats.get(feat, "") for feat in CURRENT_FEATURES])

    # Write the predictions and rationales directly to the Excel file
    predictions_df = pd.DataFrame(results)
    rationales_df = pd.DataFrame(rationale_rows)
    
    with pd.ExcelWriter(args.file, mode='a', if_sheet_exists='replace') as writer:
        predictions_df.to_excel(writer, sheet_name=args.sheet, index=False)

    print_final_usage_summary()

if __name__ == "__main__":
    main()