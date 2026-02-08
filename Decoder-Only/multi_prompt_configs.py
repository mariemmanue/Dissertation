import os
import pandas as pd
import csv
import json
import re
import time
from tqdm import tqdm
import datetime

from openai import OpenAI
import tiktoken
import argparse
import math
from transformers import pipeline as hf_pipeline
from transformers import AutoTokenizer
# from google import genai
import google.generativeai as genai
from dataclasses import dataclass
from typing import List, Dict, Any

"""
nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n phi4_gen_zs_ctx_leg_lab \
  -o Phi-4/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation/Decoder-Only && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python multi_prompt_configs.py \
    --file FullTest_Final.xlsx \
   --model microsoft/phi-4  \
   --backend phi \
    --sheet PHI4_ZS_CTX_legit_labels \
    --instruction_type zero_shot \
    --extended \
    --dialect_legitimacy \
    --context \
    --context_mode two_turn \
    --dump_prompt \
    --labels_only \
    --output_dir Phi-4/data"


"""

# Initialize global variables
total_input_tokens = 0
total_output_tokens = 0
api_call_count = 0


@dataclass
class LLMBackend:
    name: str
    model: str

    def count_tokens(self, enc_obj, messages: List[Dict[str, str]]) -> int:
        # Default: tiktoken-like encoder over concatenated contents
        text = "\n".join(m["content"] for m in messages)
        return len(enc_obj.encode(text))

    def call(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError

    

class OpenAIBackend(LLMBackend):
    def __init__(self, model: str):
        super().__init__(name="openai", model=model)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY.")
        self.client = OpenAI(api_key=api_key)

    def call(self, messages: List[Dict[str, str]]) -> str:
        resp = self.client.responses.create(
            model=self.model,
            input=messages,
        )
        return resp.output_text


class GeminiBackend(LLMBackend):
    def __init__(self, model: str):
        super().__init__(name="gemini", model=model)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY.")
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)
        self.cache = None
        self.cached_model_client = None

    def create_cache(self, system_instruction: str, model_name: str, ttl_minutes=60):
        """Creates a cached content object on Gemini servers."""
        print("Creating Gemini Context Cache...")
        try:
            from google.generativeai import caching
            
            # Create the cache
            self.cache = caching.CachedContent.create(
                model=model_name,
                display_name="aae_annotation_cache",
                system_instruction=system_instruction,
                ttl=datetime.timedelta(minutes=ttl_minutes),
            )
            
            # Create a model client specifically bound to this cache
            self.cached_model_client = genai.GenerativeModel.from_cached_content(self.cache)
            print(f"Cache created! Name: {self.cache.name}")
            return True
        except ImportError:
            print("WARNING: 'google.generativeai.caching' not found. Update library: pip install -U google-generativeai")
            return False
        except Exception as e:
            print(f"WARNING: Failed to create cache: {e}")
            return False

    def call(self, messages: List[Dict[str, str]]) -> str:
        # 1. Extract just the user message (the sentence)
        # If caching is active, 'messages' should ideally JUST be the user prompt.
        # But your 'build_messages' creates a full list. We need to be careful.
        
        user_content = ""
        # Find the last message which is the user input
        for m in reversed(messages):
            if m['role'] == 'user':
                user_content = m['content']
                break
        
        if not user_content:
            # Fallback: just join everything if we can't find a clean user block
            user_content = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

        try:
            if self.cached_model_client:
                # Use the CACHED client (System prompt is already on server)
                # We only send the user_content
                resp = self.cached_model_client.generate_content(user_content)
            else:
                # Standard legacy mode (Full text every time)
                full_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
                resp = self.client.generate_content(full_text)
                
            return resp.text
        except Exception as e:
            # Pass the exception up to query_model for retry logic
            raise e


class PhiBackend(LLMBackend):
    def __init__(self, model: str):
        super().__init__(name="phi", model=model)
        
        print(f"Loading Phi-4 from {model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, 
            trust_remote_code=True
        )
        self.pipe = hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer, # Explicitly pass tokenizer
            trust_remote_code=True,
            model_kwargs={
                "attn_implementation": "sdpa", 
                "torch_dtype": "auto", 
            },
            device_map="auto",
        )

    def count_tokens(self, enc_obj, messages: List[Dict[str, str]]) -> int:
        try:
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            return len(self.tokenizer.encode(formatted_prompt))
        except:
            text = "\n".join(m["content"] for m in messages)
            return len(self.tokenizer.encode(text))

    def call(self, messages: List[Dict[str, str]]) -> str:
        # 1. Apply Chat Template
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # 2. Generate
        outputs = self.pipe(
            prompt,
            max_new_tokens=2000, # Increased for safety with rationales
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            return_full_text=False 
        )

        generated_text = outputs[0]["generated_text"]

        # Phi-4 often wraps output in ```json ... ```
        
        # 1. Remove markdown code blocks
        if "```" in generated_text:
            # Regex to capture content inside ```json ... ``` or just ``` ... ```
            # We take the LAST block if multiple exist, or the first one found
            pattern = r"```(?:json)?\s*(.*?)```"
            matches = re.findall(pattern, generated_text, re.DOTALL)
            if matches:
                # Use the longest match or the last one (often the actual JSON)
                generated_text = matches[-1].strip()
            else:
                # Fallback: remove the markers manually if regex fails
                generated_text = generated_text.replace("```json", "").replace("```", "").strip()

        return generated_text




    def count_tokens(self, enc_obj, messages: List[Dict[str, str]]) -> int:
        # Phi-4 uses a chat template, counting raw text is inaccurate
        try:
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            return len(self.tokenizer.encode(formatted_prompt))
        except:
            # Fallback if template fails
            text = "\n".join(m["content"] for m in messages)
            return len(self.tokenizer.encode(text))

    def call(self, messages: List[Dict[str, str]]) -> str:
        # 1. Apply Chat Template (CRITICAL for Phi-4)
        # Phi-4 expects <|user|> ... <|assistant|> formatting
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # 2. Generate
        outputs = self.pipe(
            prompt,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.1,  # Lower temp = more stable for annotation
            top_p=0.9,
            return_full_text=False # Only return the new tokens
        )

        return outputs[0]["generated_text"].strip()

def extract_json_robust(text):
    # 1. Try standard cleaning & parsing first
    clean_text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    clean_text = re.sub(r'```', '', clean_text).strip()
    
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        pass # Fallthrough to regex extraction
    
    # 2. Fallback: Regex extraction of keys and values
    # This works even if the JSON is truncated at the end!
    data = {}
    
    # Pattern for simple labels: "key": 0 or "key": 1
    simple_pattern = r'"([\w-]+)":\s*(0|1)'
    for key, val in re.findall(simple_pattern, clean_text):
        data[key] = int(val)
        
    # Pattern for rationales: "key": { "value": 0, ... }
    # This is trickier, but we can try to grab the value specifically
    complex_pattern = r'"([\w-]+)":\s*\{\s*"value":\s*(0|1)'
    for key, val in re.findall(complex_pattern, clean_text):
        data[key] = int(val)
        
    if not data:
        return None # Truly failed
        
    return data

# -------------------- FEATURE LISTS --------------------

MASIS_FEATURES = [
    "zero-poss", "zero-copula", "double-tense", "be-construction", "resultant-done", "finna", "come", "double-modal",
    "multiple-neg", "neg-inversion", "n-inv-neg-concord", "aint", "zero-3sg-pres-s", "is-was-gen", "zero-pl-s",
    "double-object", "wh-qu1", "wh-qu2"
]

EXTENDED_FEATURES = [
    "zero-poss", "zero-copula", "double-tense", "be-construction", "resultant-done", "finna", "come", "double-modal",
    "multiple-neg", "neg-inversion", "n-inv-neg-concord", "aint", "zero-3sg-pres-s", "is-was-gen", "zero-pl-s",
    "double-object", "wh-qu1", "wh-qu2", "existential-it", "demonstrative-them", "appositive-pleonastic-pronoun",
    "bin", "verb-stem", "past-tense-swap", "zero-rel-pronoun", "preterite-had", "bare-got"
]

# -------------------- FEATURE BLOCKS (WITH EXAMPLES) --------------------

MASIS_FEATURE_BLOCK = """
### LIST OF AAE MORPHOSYNTACTIC FEATURES ###

### Rule 1: zero-poss
**IF** a possessive relationship is expressed WITHOUT an overt SAE possessive morpheme ('s) or standard possessive pronoun form AND this possessive meaning is clearly licensed within the same clause, **THEN** label is 1.

Typical forms:
* Noun–noun juxtaposition (dad car, mama house)
* Bare or nonstandard possessive pronouns before a noun (they car, her brother kids)

+ Example: "That they dad boo."
  * Label is 1
  * Explanation: Nonstandard 'they' + 'dad boo' expresses possession

– Example: "That's their dad's car."
  * Label is 0
  * Explanation: All possessives fully marked in SAE

Note: If it is unclear whether two adjacent nouns form a possessive relationship or just a list/name (e.g., "school bus stop"), prefer 0 and mention ambiguity.

### Rule 2: zero-copula
**IF** a form of BE (is/are/was/were) that SAE requires is missing AND the utterance can be parsed as containing a clear subject–predicate relation (not just a list, heading, or obvious subordinate fragment), **THEN** label is 1.

This applies when BE is missing:
* Before a predicate (NP, AdjP, PP), OR
* Before a V-ing progressive, OR
* Before a preverbal future/near-future marker (finna, gonna)

+ Example: "She finna eat."
  * Label is 1
  * Explanation: Missing 'is' before preverbal marker 'finna'; SAE: "She is finna eat"

– Example: "No problem."
  * Label is 0
  * Explanation: Fragment with no recoverable subject–predicate BE slot

Note: If there is a recoverable subject and predicate slot that would host BE in SAE, mark 1 even if the utterance is short or informal.

### Rule 3: double-tense
**IF** a single lexical verb shows duplicated overt past-tense morphology (usually repeated -ed) within one word, **THEN** label is 1.

+ Example: "She likeded me the best."
  * Label is 1
  * Explanation: Duplicated -ed on 'likeded'

– Example: "She liked me the best."
  * Label is 0
  * Explanation: Single past-tense marker

Note: Spelling variants that do not clearly reflect two past morphemes should be 0 unless duplication is obvious.

### Rule 4: be-construction
**IF** uninflected 'be' appears as a finite verb expressing habitual, iterative, or generalized action/state, **THEN** label is 1.

Do not mark 'be' when it functions as an auxiliary of another tense or as an agreement error.

+ Example: "They be playing outside."
  * Label is 1
  * Explanation: Habitual 'be' expressing repeated activity

– Example: "They is playing outside."
  * Label is 0
  * Explanation: Agreement generalization, not habitual 'be'

Note: If 'be' could plausibly be part of a quoting frame or a fixed phrase without clear habitual reading, prefer 0 and explain.

### Rule 5: resultant-done
**IF** preverbal 'done' directly precedes a verb phrase and marks completed aspect for that event, **THEN** label is 1.

Adverbs or discourse markers may intervene. The following verb may be past-marked or bare, but the reading must be "already/completely finished."

+ Example: "He done already ate it."
  * Label is 1
  * Explanation: 'Done' marks completed eating

– Example: "They done it yesterday."
  * Label is 0
  * Explanation: Main verb 'done', not aspect marker

Note: If 'done' can be parsed as the main verb 'did' with no clear aspectual reading, prefer 0 and note ambiguity.

### Rule 6: finna
**IF** 'finna' (or 'finta', 'fitna', etc.) functions as a preverbal marker meaning 'about to / fixing to', creating an imminent or near-future reading, **THEN** label is 1.

+ Example: "We is finna eat."
  * Label is 1
  * Explanation: 'Finna' marks imminent future

### Rule 7: come
**IF** 'come' is used as a preverbal stance/evaluative marker introducing a verb phrase or V-ing form describing someone's behavior, attitude, or speech (not as a simple motion verb), **THEN** label is 1.

+ Example: "He come talking mess again."
  * Label is 1
  * Explanation: Stance 'come' + V-ing, not motion

– Example: "He come and talked."
  * Label is 0
  * Explanation: Motion verb + coordination

Note: If 'come' can be interpreted purely as motion toward a place with no clear stance/attitude meaning, prefer 0.

### Rule 8: double-modal
**IF** two modal-like elements appear in sequence in a single clause (e.g., 'might could', 'woulda could', 'used to could') and together scope over the following verb, **THEN** label is 1.

+ Example: "I might could go."
  * Label is 1
  * Explanation: Two modals in sequence

– Example: "I might go."
  * Label is 0
  * Explanation: Single modal

### Rule 9: multiple-neg
**IF** two or more negative elements (negative auxiliaries, adverbs, pronouns, or determiners) occur within the same clause or tightly integrated predicate and together express a single semantic negation (negative concord), **THEN** label is 1.

+ Example: "I ain't never heard of that."
  * Label is 1
  * Explanation: 'Ain't' + 'never' in same clause

– Example: "I ain't ever heard of that, not anyway."
  * Label is 0
  * Explanation: Single negation; "not anyway" is scalar/emphatic, not concord

Notes:
* This is the broad category covering both negative inversion (Rule 10) and negative concord (Rule 11).
* If either Rule 10 or Rule 11 applies, this feature (multiple-neg) must also be 1.

### Rule 10: neg-inversion
**IF** a negative auxiliary or marker occurs at the beginning of a sentence or clause and precedes the subject, with the subject immediately following it in that same clause, **THEN** label is 1.

+ Example: "Don't nobody like how they actin'."
  * Label is 1
  * Explanation: Negative auxiliary 'Don't' precedes subject 'nobody'

– Example: "Nobody don't like how they actin'."
  * Label is 0
  * Explanation: Subject precedes negative auxiliary; no inversion

– Example: "I ain't never doubted nobody."
  * Label is 0
  * Explanation: Negative elements not at beginning of clause

### Rule 11: n-inv-neg-concord
**IF** both the subject and the finite verb (or auxiliary) show overt negative marking AND the subject still comes first at the beginning of a sentence or clause AND together they express a single semantic negation, **THEN** label is 1.

+ Example: "Nobody don't wanna see that."
  * Label is 1
  * Explanation: Negative subject + negative auxiliary at clause start

– Example: "Nobody wanna see that."
  * Label is 0
  * Explanation: Subject negative, verb positive

– Example: "That's how nobody never seen."
  * Label is 0
  * Explanation: Negative elements not at beginning of clause

### Rule 12: aint
**IF** 'ain't' is used as a general negative auxiliary for BE, HAVE, or DO, or as a general clausal negator (rather than as a lexical verb), **THEN** label is 1.

+ Example: "She ain't here."
  * Label is 1
  * Explanation: Negated copula

– Example: "She isn't here."
  * Label is 0
  * Explanation: Not 'ain't'

### Rule 13: zero-3sg-pres-s
**IF** a 3rd person singular subject (he, she, it, this/that NP, nobody, somebody, etc.) co-occurs with a bare verb or uninflected auxiliary (do, have, walk, go) in PRESENT-TENSE meaning where SAE would require -s or does/has, **THEN** label is 1.

Exclude non-agreeing 'is/was' forms (those are Rule 14: is-was-gen).

+ Example: "She walk to they house."
  * Label is 1
  * Explanation: 3sg subject + bare 'walk' with present meaning; SAE: 'walks'

– Example: "They walk to their house."
  * Label is 0
  * Explanation: Plural subject; no 3sg requirement

Note: **If the bare verb occurs in reported speech, restarts, conditional/subjunctive context, or embedded clauses where the syntactic environment is ambiguous, prefer 0.** Only mark when the clause is clearly a finite present-tense declarative with 3sg subject.

### Rule 14: is-was-gen
**IF** 'is' or 'was' is used in a way that ignores SAE person/number agreement (e.g., with plural or 1st person subjects) in a finite clause, **THEN** label is 1.

Do NOT mark for existential 'it' constructions ("It was a fight," "It's people out here"), which are grammatical in SAE.

+ Example: "They was there."
  * Label is 1
  * Explanation: Plural subject + 'was'

– Example: "He was there."
  * Label is 0
  * Explanation: SAE-agreeing

Note: If 'was' may be part of a quoting frame or reported speech with unclear subject, prefer 0.

### Rule 15: zero-pl-s
**IF** a noun that clearly has plural reference (from a quantifier, determiner, or context) surfaces without SAE plural -s AND the plural reading is local to the noun phrase, **THEN** label is 1.

+ Example: "She got them dog."
  * Label is 1
  * Explanation: Plural demonstrative 'them' + bare 'dog'

– Example: "A dogs."
  * Label is 0
  * Explanation: Article–noun mismatch, not AAE plural pattern

Note: If plurality is only inferable from distant context and not clear in the NP itself, prefer 0.

### Rule 16: double-object
**IF** in a single clause, the subject and a following object pronoun (me, us, you, him, her, them) are coreferential (e.g., I…me, we…us, you…you) AND that pronoun is immediately followed by a noun phrase (NP) with no preposition (no to/for) AND together the verb + pronoun + NP express that the subject is obtaining, having, or wanting something for themself, **THEN** label is 1.

+ Example: "We had us a couple of beers."
  * Label is 1
  * Explanation: Subject 'we' = pronoun 'us'; self-benefactive

+ Example: "Soon as you get you some Scrabble tiles…"
  * Label is 1
  * Explanation: Subject 'you' = pronoun 'you'; self-benefit

– Example: "He gave me a book."
  * Label is 0
  * Explanation: Pronoun 'me' not coreferential with subject 'he'

– Example: "They got him a car."
  * Label is 0
  * Explanation: Subject 'they' ≠ pronoun 'him'

Notes:
* The crucial diagnostic is subject = object pronoun and a following NP with no preposition.
* Do not mark ordinary ditransitives where the pronoun refers to a different person.
* Exclude cases where the second element after the pronoun is not a full NP (e.g., tell you what, show you how).

### Rule 17a: wh-qu1 (WH-word + zero copula/DO deletion)
**IF** the string is a genuine WH-interrogative that makes a direct request for information AND SAE would require a form of BE or DO in that question, **THEN** label is 1.

This includes:
* Zero copula before a predicate or locative (Where she at?)
* Missing DO in WH-questions (What you want?, Where you go?)

+ Example: "Who you be talking to like that?"
  * Label is 1
  * Explanation: Missing 'are' between wh-word and subject; SAE: 'Who are you usually talking to like that?'

+ Example: "What you want?"
  * Label is 1
  * Explanation: Missing 'do'

– Example: "Where is she?"
  * Label is 0
  * Explanation: Auxiliary present

– Example: "I don't know what she wants."
  * Label is 0
  * Explanation: Not requesting information

Notes:
* Mark BOTH wh-qu1 AND zero-copula when the missing auxiliary is required for a WH-question in SAE.
* Do not mark for complements, fragments, or subordinate what/where clauses that are not clearly questions.

### Rule 17b: wh-qu2 (WH-word + non-standard inversion)
**IF** a WH-question or WH-clause departs from SAE subject–auxiliary inversion patterns (no inversion where SAE requires it OR inversion inside embedded WH-clauses where SAE keeps declarative order), **THEN** label is 1.

+ Example: "Where he is going?"
  * Label is 1
  * Explanation: Auxiliary follows subject in a main question

– Example: "I asked him if he could find her."
  * Label is 0
  * Explanation: Not a wh-question; standard word order

Note: Only mark wh-qu2 when WH-clause word order is non-standard relative to SAE.
"""

NEW_FEATURE_BLOCK = MASIS_FEATURE_BLOCK + """

### Rule 18: existential-it
**IF** 'it' functions as an existential/dummy subject in a construction where SAE would normally use 'there' to introduce an existential AND the following predicate introduces new entities, **THEN** label is 1.

+ Example: "It's people out here don't care."
  * Label is 1
  * Explanation: Existential 'it' + people; SAE: 'There are people'

– Example: "It is raining out here."
  * Label is 0
  * Explanation: Weather 'it' is grammatical in SAE

Note: If 'it' can be read as a true referential pronoun with a clear antecedent, prefer 0.

### Rule 19: demonstrative-them
**IF** 'them' is used directly before a noun as a demonstrative determiner meaning 'those' (not as an object pronoun), **THEN** label is 1.

+ Example: "Them shoes tight."
  * Label is 1
  * Explanation: 'Them' functions as demonstrative determiner

+ Example: "All them people."
  * Label is 1
  * Explanation: 'Them' as demonstrative even without immediately adjacent noun

– Example: "I like them."
  * Label is 0
  * Explanation: Object pronoun

+ Example: "See all them over there."
  * Label is 1
  * Explanation: 'Them' as demonstrative even when noun is ellipted

Note: 'Them' counts as demonstrative when it precedes a noun (even if a quantifier like 'all' intervenes) or when the noun is clearly recoverable from context (ellipted noun).


### Rule 20: appositive-pleonastic-pronoun
**IF** a subject or object NP is followed, **within the same clause**, by a clearly co-referential pronoun **in the same grammatical role** (subject + subject OR object + object), forming an appositive or pleonastic structure, **THEN** label is 1.

The pronoun must be **redundant**—the clause would be grammatical in SAE without it.

Fillers or pauses (e.g., 'uh', 'you know') may appear between NP and pronoun.

+ Example: "My dad, he told me it."
  * Label is 1
  * Explanation: NP 'my dad' + resumptive subject 'he' in same clause; clause would be grammatical as "My dad told me it"

+ Example: "The lawyer, I forgot about him."
  * Label is 0
  * Explanation: 'The lawyer' is left-dislocated topic; 'him' is the required object of "forgot about," not redundant

– Example: "A lot of people, you can tell they would tell me that."
  * Label is 0
  * Explanation: 'You' is subject of different clause; 'they' is also in different clause

Note: The key diagnostic is whether removing the pronoun leaves a grammatical SAE clause. **Do not mark complex NPs with nested modifiers as appositive-pleonastic unless there is a separate, co-referential pronoun in the same grammatical role within the same clause.** If the pronoun is required as subject or object, it is not pleonastic.


### Rule 21: bin
**IF** BIN/been appears without an overt auxiliary 'have' AND expresses a long-standing or remote past state, **THEN** label is 1.

+ Example: "She been married."
  * Label is 1
  * Explanation: Long-standing state without 'have'

– Example: "She's been married for two years."
  * Label is 0
  * Explanation: Standard 'have been'

### Rule 22: verb-stem
**IF** a bare (uninflected) verb form serves as the finite verb of a clause that clearly refers to a past event based on local cues (explicit time adverbs, nearby past-tense anchors, or aspect markers like 'done') where SAE would require a past-tense form, **THEN** label is 1.

Local past-time cues include:
* Explicit time adverbs (yesterday, last week, ago)
* A past-tense verb in a coordinate or serial construction (e.g., "He ran and jump" – 'ran' anchors 'jump' as past)
* Aspect markers like 'done' that establish completed past

+ Example: "Yesterday he done walk to school."
  * Label is 1
  * Explanation: Bare 'walk' in past context; SAE: 'walked'

– Example: "He walk to school every day."
  * Label is 0
  * Explanation: Present habitual; possible zero-3sg-pres-s but not verb-stem

Note: **In coordinate or serial verb constructions (comma-separated verbs describing a sequence of events), if either verb is past-tense and another verb is bare, the bare verb counts as verb-stem because the first verb anchors the time reference as past.** If there is no explicit evidence that the event is past, prefer 0 and avoid assuming past meaning.

### Rule 23: past-tense-swap
**IF** an overtly non-SAE tense form is used as the main tense carrier of a clause AND the clause has clear simple-past or perfect/pluperfect reference, **THEN** label is 1.

Mark 1 when:
* A past participle is used as simple past (e.g., 'seen', 'done' for 'saw', 'did'), OR
* A regularized past is used where SAE requires irregular (e.g., 'throwed' for 'threw'), OR
* A past-tense form appears in any position where SAE requires a non-tensed form (bare infinitive after 'do/did/does', or distinct participle after 'have/had')

+ Example: "I seen him yesterday."
  * Label is 1
  * Explanation: Past participle 'seen' used as preterite; SAE: 'saw'

+ Example: "The dog had bit him before."
  * Label is 1
  * Explanation: Simple past 'bit' in pluperfect position; SAE: 'had bitten'

– Example: "I saw him yesterday."
  * Label is 0
  * Explanation: Standard preterite form

Note: This feature never applies to bare stems (those are verb-stem). If the verb has no overt tense morphology, mark 0 for past-tense-swap.

### Rule 24: zero-rel-pronoun
**IF** a finite clause modifies a noun and functions as a subject relative AND there is NO overt relative pronoun ('who', 'that', 'which') in subject position, **THEN** label is 1.

+ Example: "There are many mothers don't know their children."
  * Label is 1
  * Explanation: Clause modifying 'mothers' without 'who'

– Example: "I think he left."
  * Label is 0
  * Explanation: That-deletion in complement clause, not subject relative

### Rule 25: preterite-had
**IF** 'had' plus a past verb is used to express a simple past event (with no clear 'past-before-past' meaning) AND there is no later past event anchoring a pluperfect reading, **THEN** label is 1.

Often appears with regularized/AAE-style past forms (had went, had ran).

+ Example: "The alarm next door had went off a few minutes ago."
  * Label is 1
  * Explanation: Simple past meaning; no later reference event

– Example: "They had seen the movie before we arrived."
  * Label is 0
  * Explanation: True pluperfect (past-before-past)

Note: If there is a clear second past event that makes 'had' plausibly pluperfect, prefer 0 and treat as SAE pluperfect.

### Rule 26: bare-got
**IF** 'got' functions as a present-tense possessive verb meaning 'have' AND there is NO overt 'have/has' or 'have got' construction in the same clause, **THEN** label is 1.

The subject has something right now (current possession), not a past 'got' event.

+ Example: "I got three kids."
  * Label is 1
  * Explanation: Present possession; SAE: 'I have three kids'

– Example: "I got a big paycheck last month."
  * Label is 0
  * Explanation: Simple past of 'get', not present possession

Note: Only mark bare-got when the clause clearly describes current possession and does not contain an overt 'have/has'. If 'got' can be read as simple past or as part of 'have got', prefer 0.
"""

# -------------------- ICL FEW-SHOT EXAMPLES BLOCK --------------------

ICL_EXAMPLES_BLOCK = """
### FEW-SHOT TRAINING EXAMPLES ###
(for demonstration only; NOT the target utterance)

**Example 1:**
SENTENCE: "And my cousin family place turn into a whole cookout soon as it get warm, and when you step outside it's people dancing out on the sidewalk."

ANNOTATED LABELS (subset):
- zero-3sg-pres-s: {
    "value": 1,
    "rationale": "In 'my cousin family place turn into a whole cookout', the subject is 3sg ('my cousin family place') and the present-tense verb 'turn' lacks -s; SAE requires 'turns', so label is 1."
}
- existential-it: {
    "value": 1,
    "rationale": "In 'it's people dancing out on the sidewalk', 'it' is a dummy subject that introduces new people, like 'There are people dancing', so label is 1."
}
- multiple-neg: {
    "value": 0,
    "rationale": "No clause in the sentence contains more than one negative element, so label is 0."
}
- zero-poss: {
    "value": 1,
    "rationale": "In 'my cousin family place', the nouns 'cousin' and 'family' express possession of 'place' without an overt 's (SAE: 'my cousin's family place'), so label is 1."
}

**Example 2:**
SENTENCE: "He throwed him a quick punch, then spin around and walk straight out the room like it was nothing."

ANNOTATED LABELS (subset):
- double-object: {
    "value": 0,
    "rationale": "In 'throwed him a quick punch', the verb is followed by two NP objects but this frame is standard for SAE ditransitive, so label is 0."
}
- verb-stem: {
    "value": 1,
    "rationale": "Past time is set by 'He throwed him a quick punch'. The later verbs 'spin' and 'walk' are bare stems in the same past-time sequence where SAE expects 'spun' and 'walked', so label is 1."
}
- past-tense-swap: {
    "value": 1,
    "rationale": "'Throwed' is a regularized past form used as simple past where SAE uses irregular 'threw', so label is 1."
}

**Example 3:**
SENTENCE: "He ain't never seen nothing move that fast before, then a week later he just seen it happen again right in front of him."

ANNOTATED LABELS (subset):
- multiple-neg: {
    "value": 1,
    "rationale": "In 'ain't never seen nothing', the negative elements 'ain't', 'never', and 'nothing' are all in the same clause and together mean 'has never seen anything', so label is 1."
}
- aint: {
    "value": 1,
    "rationale": "'Ain't' is used as a negative auxiliary for HAVE ('hasn't seen'), not as a main verb, so label is 1."
}
- past-tense-swap: {
    "value": 1,
    "rationale": "In 'he just seen it happen again', 'seen' (a participle form) is used as the only past-tense form where SAE requires 'saw', so label is 1."
}
- verb-stem: {
    "value": 0,
    "rationale": "No bare stem verb serves as the finite past predicate: 'seen' is a participle used as simple past (captured by past-tense-swap), and 'happen' is non-finite in 'seen it happen', so label is 0."
}

Use these as patterns: for each feature, look for the exact words that match the decision rule and base your 0/1 decision on that local evidence only.
Do NOT reuse these sentences when analyzing the new target utterance.
"""

# -------------------- UTILITIES --------------------

def drop_features_column(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=['FEATURES', 'Source'], errors='ignore')

def strip_examples_from_block(block: str) -> str:
    """
    Remove example lines from a feature block to create a rules-only version.
    Used for zero-shot / zero-shot CoT conditions.
    """
    stripped_lines = []
    for line in block.splitlines():
        s = line.lstrip()
        # Skip lines that start with + or – followed by Example/Miss
        if s.startswith("+ Example") or s.startswith("– Example"):
            continue
        # Skip bullet explanations under examples (lines starting with * after examples)
        if s.startswith("* Label is") or s.startswith("* Explanation:"):
            continue
        stripped_lines.append(line)
    return "\n".join(stripped_lines)

def build_system_msg(
    instruction_type: str,
    dialect_legitimacy: bool,
    self_verification: bool,
    use_context: bool,  
) -> dict:
    """
    Build the system message with improved structure.
    """
    
    if dialect_legitimacy:
        intro = (
            "You are a highly experienced sociolinguist and expert annotator of African American English (AAE).\n"
            "AAE is a rule-governed, systematic language variety. You must analyze the input according to AAE's "
            "internal grammatical rules, not Standard American English (SAE) norms.\n"
            "Treat African American English and Standard American English as equally valid.\n"
            "Do not 'correct' or rate sentences. Your only task is to decide, for each feature, whether its definition matches this utterance (1) or not (0).\n"
            "Informal or slangy English that could be spoken by many speakers (regardless of race) does NOT automatically "
            "count as AAE. Only mark a feature as 1 if the utterance clearly matches the specific AAE morphosyntactic "
            "pattern described in the decision rule.\n\n"
        )
    else:
        intro = (
            "You are a linguist analyzing morphosyntactic features in a variety of English often referred to as "
            "African American English (AAE).\n"
            "Your goal is to identify specific grammatical constructions in the input utterance, comparing them to "
            "Standard American English (SAE) where relevant.\n\n"
        )

    base_content = (
        intro +
        "Your task is to make strictly binary decisions (1 = present, 0 = absent) about a fixed list of AAE "
        "morphosyntactic features for a single target utterance.\n\n"
        "### PROCEDURE ###\n"
        "Follow these steps, keeping explanations concise:\n\n"
        "**1. CLAUSE & TENSE ANALYSIS (global):**\n"
        "* Identify the main clause of the TARGET UTTERANCE\n"
        "* Identify its grammatical subject(s), finite verb(s), and any auxiliaries\n"
        "* Identify tense/aspect markers (e.g., done, BIN, had) and overt temporal expressions (yesterday, last week)\n"
        "* Identify the scope and pattern of negation and any embedded clauses\n\n"
        "**2. FEATURE-BY-FEATURE EVALUATION:**\n"
        "* For each feature, check whether the TARGET UTTERANCE satisfies ALL parts of that rule\n"
        "* For tense-related features, ask:\n"
        "  - What tense/aspect form does the verb show? (bare, simple past, past participle, etc.)\n"
        "  - What form does SAE require in this syntactic position? (after auxiliary 'did', after 'have/had', as main finite verb, etc.)\n"
        "  - Does the mismatch fit the AAE pattern described in the rule?\n"
        "* Focus each decision on that single element\n\n"

        "**3. MULTIPLE INTERPRETATIONS:**\n"
        "* If more than one grammatical analysis is genuinely possible, briefly acknowledge the strongest alternative\n"
        "* Then choose the label (1 or 0) that best satisfies the feature's explicit decision rule\n"
        "* **Prefer precision over recall**: if the utterance does not provide enough grammatical evidence to confidently apply the rule, output 0 and explain that the structure is underspecified\n"
        "* Do NOT treat disfluencies, missing subjects, or casual phrasing as ambiguity unless they obscure the syntactic environment relevant to the feature\n\n"
    )

    if self_verification:
        base_content += (
            "**4. SELF-VERIFICATION (FINAL CHECK BEFORE OUTPUT):**\n"
            "For each feature, confirm that:\n"
            "* Your reasoning uses ONLY grammatical facts in the utterance—word order, morphology, clause structure, tense/aspect\n"
            "* You are NOT filling in missing material with world knowledge or assumptions about intent\n"
            "* You are NOT labeling a structure as AAE merely because it differs from SAE or sounds informal\n"
            "* Your rationale explains why the rule either applies or does not apply, even if the label is 0\n\n"
        )

    base_content += (
        "### EXPLICIT EVALUATION CONSTRAINTS ###\n"
        "* Analyze ONLY syntax, morphology, and clause structure\n"
        "* Base each decision only on the feature definitions and the words in the target utterance (and context, when explicitly allowed)\n"
        "* Do not infer extra events or repair the sentence\n"
        "* Informal, conversational, or slang does NOT by itself imply an AAE feature\n"
        "* Only mark a feature when the specific AAE decision rule is satisfied\n\n"
        "**SUBJECT DROPS IN SPONTANEOUS SPEECH:**\n"
        "* In conversational speech, speakers often omit subjects when pragmatically given\n"
        "* If a clause has a clearly recoverable subject from the SAME utterance or immediately preceding clause, you may treat that subject as syntactically present when relevant\n"
        "* Compare the verb form to what SAE would require with that understood subject\n"
        "* If no specific subject is clearly recoverable, prefer 0 and note that the structure is too fragmentary\n\n"
        "**FOR TENSE-RELATED FEATURES:**\n"
        "* First determine the intended reference time (past vs present vs habitual) using explicit time adverbs, aspect markers, and local discourse context\n"
        "* Then compare the verb form to what SAE would require in that same context\n"
        "* Only mark the feature as 1 when the mismatch fits the defined AAE pattern\n\n"
    )

    if use_context:
        base_content += (
            "**CONTEXT USE (if provided):**\n"
            "* You may use PREV/NEXT sentence context ONLY to resolve:\n"
            "  - Recoverable subject in the SAME utterance chain\n"
            "  - Explicit temporal reference (past vs present) when stated in context\n"
            "* Do NOT use context to invent missing words or infer events not stated\n\n"
        )

    if "cot" in instruction_type:
        base_content += (
            "### ADDITIONAL REASONING REQUIREMENT (CHAIN-OF-THOUGHT) ###\n"
            "* Before filling the JSON, internally reason step by step about clause structure, tense/aspect, negation, and the single semantic element for each feature\n"
            "* Do NOT output those intermediate steps separately; only include final binary values (and rationales, if requested) in the JSON\n\n"
        )

    return {"role": "system", "content": base_content}

def build_system_response_instructions(
    utterance: str,
    features: list[str],
    instruction_type: str,
    require_rationales: bool,
    context_block: str | None = None,
) -> str:
    feature_list_str = ", ".join(f'"{f}"' for f in features)

    if require_rationales:
        output_format = (
            "### OUTPUT FORMAT (STRICT) ###\n"
            "* Return ONLY a single JSON object\n"
            "* It MUST contain EXACTLY these keys, in this order\n"
            "* For each key, provide an object with:\n"
            "  - 'value': 1 or 0 (integer, not string)\n"
            "  - 'rationale': 1–2 sentences of grammatical reasoning\n\n"
            "The rationale SHOULD:\n"
            "* Point to the exact substring that justifies 1 (e.g., bare verb with past adverb, duplicated tense, WH + missing copula)\n"
            "* Or explicitly state that no such substring exists for 0\n\n"
            "Example output structure:\n"
            "{\n"
            '  "zero-poss": {"value": 0, "rationale": "No bare possessive noun–noun sequence; possessive is fully marked in SAE form."},\n'
            '  "zero-copula": {"value": 1, "rationale": "Subject is followed by predicate adjective with no overt form of BE in a single clause."},\n'
            "  ...\n"
            f'  "{features[-1]}": {{"value": 0, "rationale": "Word order in the WH-clause matches SAE, so this WH feature does not apply."}}\n'
            "}\n\n"
            "Do NOT add extra fields. Do NOT change key names. Do NOT include explanation outside this JSON.\n"
        )
    else:
        output_format = (
            "### OUTPUT FORMAT (STRICT, LABEL-ONLY) ###\n"
            "* Return ONLY a single JSON object\n"
            "* It MUST contain EXACTLY these keys, in this order\n"
            "* For each key, the value MUST be exactly 1 or 0 (integer, not string)\n"
            "* Do NOT include nested objects, rationales, or extra fields\n\n"
            "Example output structure:\n"
            "{\n"
            '  "zero-poss": 0,\n'
            '  "zero-copula": 1,\n'
            "  ...\n"
            f'  "{features[-1]}": 0\n'
            "}\n\n"
            "Do NOT add extra fields. Do NOT change key names. Do NOT include explanation outside this JSON.\n"
        )

    context_section = ""
    if context_block:
        context_section = (
            "### CONTEXT ###\n"
            "(use only as permitted by system instructions)\n\n"
            f"{context_block}\n\n"
        )

    return (
        "Now analyze the target utterance strictly according to the rules above.\n\n"
        f"{context_section}"
        f"**UTTERANCE (target):** {utterance}\n\n"
        "### TASK ###\n"
        "* For EACH feature in the list below, decide whether it is present (1) or absent (0) IN THIS UTTERANCE\n"
        "* Treat each feature as a separate task-grounded question about ONE semantic/grammatical element\n"
        "* Apply the procedure before making each binary decision\n\n"
        f"**FEATURE KEYS (in required order):**\n[{feature_list_str}]\n\n"
        f"{output_format}"
    )

def has_usable_context(left_context: str | None, right_context: str | None) -> bool:
    return bool(_clean_ctx(left_context) or _clean_ctx(right_context))

def _clean_ctx(x):
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    s = str(x).strip()
    return s if s else None

def format_context_block(left_context: str | None, right_context: str | None) -> str:
    left_context = _clean_ctx(left_context)
    right_context = _clean_ctx(right_context)

    lines = []
    if left_context:
        lines.append(f"PREV SENTENCE (context): {left_context}")
    if right_context:
        lines.append(f"NEXT SENTENCE (context): {right_context}")
    return "\n".join(lines)



def build_messages(
    utterance: str,
    features: list[str],
    base_feature_block: str,
    instruction_type: str,
    include_examples_in_block: bool,
    use_context: bool = False,
    left_context: str | None = None,
    right_context: str | None = None,
    context_mode: str = "single_turn",   # <-- NEW
    use_icl_examples: bool = False,
    dialect_legitimacy: bool = False,
    self_verification: bool = False,
    require_rationales: bool = True,
) -> tuple[list[dict], str]:

    system_msg = build_system_msg(
        instruction_type=instruction_type,
        dialect_legitimacy=dialect_legitimacy,
        self_verification=self_verification,
        use_context=use_context,
    )

    effective_include_examples = include_examples_in_block
    if instruction_type in ["zero_shot", "zero_shot_cot"]:
        effective_include_examples = False

    feature_block = base_feature_block if effective_include_examples else strip_examples_from_block(base_feature_block)

    # Build context block text once
    context_block = None
    if use_context:
        cb = format_context_block(left_context, right_context)
        if cb.strip():
            context_block = cb

    # === Condition B: context in its own earlier user message ===
    if use_context and context_block and context_mode == "two_turn":
        context_msg = {
            "role": "user",
            "content": (
                "CONTEXT (do NOT analyze yet; this is NOT the target utterance).\n"
                "Use only as permitted by the system instructions.\n\n"
                f"{context_block}"
            ),
        }

        # Main task message contains NO embedded context
        parts = [feature_block]
        if use_icl_examples:
            parts.append(ICL_EXAMPLES_BLOCK)

        response_instructions = build_system_response_instructions(
            utterance=utterance,
            features=features,
            instruction_type=instruction_type,
            require_rationales=require_rationales,
            context_block=None,  # important: keep empty here for the two-turn condition
        )
        parts.append(response_instructions)

        task_msg = {"role": "user", "content": "\n\n".join(parts)}
        return [system_msg, context_msg, task_msg], "two_turn"

    # === Condition A (default): single user message (your current approach) ===
    parts = [feature_block]
    if use_icl_examples:
        parts.append(ICL_EXAMPLES_BLOCK)

    response_instructions = build_system_response_instructions(
        utterance=utterance,
        features=features,
        instruction_type=instruction_type,
        require_rationales=require_rationales,
        context_block=context_block if (use_context and context_block) else None,
    )
    parts.append(response_instructions)

    user_msg = {"role": "user", "content": "\n\n".join(parts)}
    
    return [system_msg, user_msg], "single_turn"
# -------------------- USAGE SUMMARY --------------------

def print_final_usage_summary():
    total_tokens = total_input_tokens + total_output_tokens

    print("\n===== FINAL USAGE SUMMARY =====")
    print(f"Total API Calls: {api_call_count}")
    print(f"Total Input Tokens: {total_input_tokens}")
    print(f"Total Output Tokens: {total_output_tokens}")
    print(f"Total Tokens: {total_tokens}")
    print("===================================\n")


def _extract_first_json_object(text: str) -> str | None:
    """
    Best-effort: find the first top-level {...} JSON object in a messy string.
    Handles leading text, code fences, etc. Does NOT fix invalid JSON (e.g., single quotes).
    """
    if not text:
        return None

    # Strip common code fences quickly
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
    return None


def parse_output_json(raw_str: str, features: list[str]):
    # Use the robust function we defined earlier
    data = extract_json_robust(raw_str)
    
    if data is None:
        raise ValueError("Failed to extract JSON")

    
    if data is None:
        # Fallback: Treat as total failure or raise exception
        # You can raise exception to trigger the "PARSEFAIL" logic in main()
        raise ValueError("Failed to extract JSON from model output")

    vals = {}
    rats = {}
    missing = []
    for feat in features:
        if feat not in data:
            missing.append(feat)

        entry = data.get(feat, {})
        if isinstance(entry, (int, float)):
            vals[feat] = int(entry)
            rats[feat] = ""
        elif isinstance(entry, dict):
            vals[feat] = int(entry.get("value", 0))
            rats[feat] = str(entry.get("rationale", "")).strip()
        else:
            vals[feat] = 0
            rats[feat] = ""



    return vals, rats, missing



# -------------------- GPT QUERY --------------------

def query_model(
    backend: LLMBackend,
    enc_obj,
    sentence: str,
    features: list[str],
    base_feature_block: str,
    instruction_type: str,
    include_examples_in_block: bool,
    use_context: bool = False,
    left_context: str | None = None,
    right_context: str | None = None,
    context_mode: str = "single_turn",
    dialect_legitimacy: bool = False,
    self_verification: bool = False,
    require_rationales: bool = True,
    dump_prompt: bool = False,
    dump_prompt_path: str | None = None,
    dump_once_key: str | None = None,
    max_retries: int = 15,
    base_delay: int = 3,
) -> tuple[str | None, str]:
    global api_call_count, total_input_tokens, total_output_tokens

    use_icl_examples = instruction_type in ["icl", "few_shot_cot"]

    messages, arm_used = build_messages(
        utterance=sentence,
        features=features,
        base_feature_block=base_feature_block,
        instruction_type=instruction_type,
        include_examples_in_block=include_examples_in_block,
        use_context=use_context,
        context_mode=context_mode,
        left_context=left_context,
        right_context=right_context,
        use_icl_examples=use_icl_examples,
        dialect_legitimacy=dialect_legitimacy,
        self_verification=self_verification,
        require_rationales=require_rationales,
    )
    if isinstance(backend, GeminiBackend) and backend.cached_model_client is not None:
        messages = [m for m in messages if m['role'] != 'system']
        
    if dump_prompt and dump_once_key and os.environ.get(dump_once_key, "0") != "1":
        os.environ[dump_once_key] = "1"
        payload = {"backend": backend.name, "model": backend.model, "messages": messages}
        print("\n===== PROMPT DUMP =====")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("===== END PROMPT DUMP =====\n")
        if dump_prompt_path:
            with open(dump_prompt_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

    input_tokens = backend.count_tokens(enc_obj, messages)
    total_input_tokens += input_tokens

    for attempt in range(max_retries):
        try:
            output_text = backend.call(messages)
            output_tokens = len(enc_obj.encode(output_text))
            total_output_tokens += output_tokens
            api_call_count += 1
            print(output_text)
            print(
                f"API Call #{api_call_count} | "
                f"Input Tokens: {input_tokens} | "
                f"Output Tokens: {output_tokens} | "
                f"Total: {total_input_tokens + total_output_tokens}"
            )
            return output_text, arm_used
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate" in msg.lower():
                wait_time = base_delay * (2 ** attempt)
                print(f"Rate limit hit. Retrying in {wait_time}s... (Attempt {attempt + 1})")
                time.sleep(wait_time)
            else:
                print(f"Error on sentence: {sentence[:40]}... = {e}")
                return None, arm_used

    print(f"Failed after {max_retries} retries.")
    return None, arm_used


# -------------------- MAIN --------------------
def main():
    import torch
    print(f"DEBUG: torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"DEBUG: torch.version.cuda = {torch.version.cuda}")

    
    
    dumponcekey = "DUMPED_PROMPT_ONCE"

    parser = argparse.ArgumentParser(description="Run GPT Experiments for AAE feature annotation.")
    parser.add_argument("--file", type=str, help="Input Excel file path", required=True)
    parser.add_argument("--sheet", type=str, help="Sheet name to write GPT predictions into", required=True)
    parser.add_argument("--extended", action="store_true", help="Use extended feature set (NEW_FEATURE_BLOCK / EXTENDED_FEATURES)")
    parser.add_argument("--context", action="store_true", help="Use prev/next sentence context from the same sheet (neighbor rows).")
    parser.add_argument(
        "--instruction_type",
        type=str,
        choices=["zero_shot", "icl", "zero_shot_cot", "few_shot_cot"],
        default="zero_shot",
        help="Instruction style: zero_shot, icl (few-shot ICL), zero_shot_cot, few_shot_cot",
    )
    parser.add_argument(
        "--block_examples",
        action="store_true",
        help="If set, keep 'Example' / '–' lines inside the feature block (non-zero-shot conditions only).",
    )
    parser.add_argument(
        "--dialect_legitimacy",
        action="store_true",
        help="If set, explicitly frame AAE as rule-governed and legitimate, and instruct the model not to treat AAE as errors.",
    )
    parser.add_argument(
        "--self_verification",
        action="store_true",
        help="If set, include an explicit self-verification step in the system instructions.",
    )
    parser.add_argument(
        "--labels_only",
        action="store_true",
        help="If set, request flat JSON labels (0/1) only with NO rationales.",
    )
    parser.add_argument("--output_dir", type=str, help="Output directory for CSV/Excel results", required=True)
    parser.add_argument(
        "--dump_prompt",
        action="store_true",
        help="Print the exact messages payload sent to the API (first example only).",
    )
    parser.add_argument(
        "--dump_prompt_path",
        type=str,
        default=None,
        help="If set, also write the exact prompt payload to this path as JSON.",
    )
    parser.add_argument(
        "--context_mode",
        type=str,
        choices=["single_turn", "two_turn"],
        default="single_turn",
        help="Context mode (kept for compatibility with build_messages).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["openai", "gemini", "phi"],
        default="openai",
        help="LLM backend to use.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5",
        help="Model name for the chosen backend (e.g., gpt-5, gemini-2.0-flash, phi-4).",
    )


    args = parser.parse_args()

    file_title = os.path.splitext(os.path.basename(args.file))[0]
    outdir = os.path.join(args.output_dir, file_title)
    os.makedirs(outdir, exist_ok=True)

    # META CSV path
    metapath = os.path.join(outdir, args.sheet + "_meta.csv")
    if not os.path.exists(metapath):
        with open(metapath, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    "idx",
                    "sentence",
                    "use_context_requested",
                    "has_usable_context",
                    "requested_context_mode",
                    "arm_used",
                    "context_included",
                    "parse_status",
                    "missing_key_count",
                    "missing_keys",
                ]
            )

    def writemeta(idx, sentence, usable, arm_used, context_included, parse_status, missing_count, missingkeys):
        with open(metapath, "a", newline="", encoding="utf-8") as f:
            use_ctx_req = int(args.context)
            requested = args.context_mode if args.context else ""
            csv.writer(f).writerow(
                [
                    idx,
                    sentence,
                    use_ctx_req,
                    int(usable),
                    requested,
                    arm_used,
                    int(bool(context_included)),
                    parse_status,
                    missing_count,
                    missingkeys,
                ]
            )

    # -------------------- LOAD DATA --------------------
    sheets = pd.read_excel(args.file, sheet_name=None)
    # Assuming your Gold sheet is named "Gold" as in the original pipeline
    golddf = sheets["Gold"]
    golddf = golddf.dropna(subset=["sentence"]).reset_index(drop=True)

    eval_sentences = golddf["sentence"].dropna().tolist()
    print(f"Number of sentences to evaluate: {len(eval_sentences)}")

    if args.backend == "openai":
        backend = OpenAIBackend(model=args.model)
        try:
            enc = tiktoken.encoding_for_model(args.model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
    elif args.backend == "gemini":
        backend = GeminiBackend(model=args.model)
        # No official Gemini tokenizer; use a stable approximate
        enc = tiktoken.get_encoding("cl100k_base")
    elif args.backend == "phi":
        backend = PhiBackend(model=args.model)
        # enc will be ignored by PhiBackend.count_tokens, but set something to satisfy the interface
        enc = tiktoken.get_encoding("cl100k_base")
    else:
        raise ValueError(f"Unknown backend {args.backend}")


    # -------------------- CLIENT SETUP --------------------
    # openai_api_key = os.getenv("OPENAI_API_KEY")
    # if not openai_api_key:
    #     raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    # tokenizer
    # enc = tiktoken.encoding_for_model(args.model if args.backend == "openai" else "gpt-4o-mini")  # or a default

    if args.backend == "openai":
        backend = OpenAIBackend(model=args.model)
    elif args.backend == "gemini":
        backend = GeminiBackend(model=args.model)
    elif args.backend == "phi":
        backend = PhiBackend(model=args.model)
    else:
        raise ValueError(f"Unknown backend {args.backend}")

    use_extended = args.extended
    CURRENT_FEATURES = EXTENDED_FEATURES if use_extended else MASIS_FEATURES
    BASE_FEATURE_BLOCK = NEW_FEATURE_BLOCK if use_extended else MASIS_FEATURE_BLOCK

    include_examples_in_block = args.block_examples
    require_rationales = not args.labels_only

    preds_path = os.path.join(outdir, args.sheet + "_predictions.csv")
    rats_path = os.path.join(outdir, args.sheet + "_rationales.csv")

    # -------------------- RESUME SUPPORT: LOAD EXISTING OUTPUTS --------------------
    def get_resume_idxs(preds_path, eval_sentences):
        """Robust resume: row count proxy + last sentence verification"""
        if not os.path.exists(preds_path):
            print(f"Debug: No predictions file at {preds_path}")
            return set()
        
        try:
            existing_df = pd.read_csv(preds_path)
            print(f"Debug: Columns found: {list(existing_df.columns)}")
            print(f"Debug: {len(existing_df)} rows in existing file")
            
                # Strategy 1: Explicit 'idx' column (The preferred way now)
            if 'idx' in existing_df.columns:
                return set(existing_df['idx'].tolist())
        
            # Strategy 2: Row count proxy + LAST SENTENCE VERIFICATION
            if 'sentence' in existing_df.columns and len(existing_df) > 0:
                num_rows = len(existing_df)
                resume_idx_candidate = num_rows
                
                # Verify LAST sentence matches expected position
                if len(eval_sentences) > num_rows - 1:
                    last_csv_sentence = str(existing_df.iloc[-1]['sentence']).strip().lower()
                    expected_sentence_idx = num_rows - 1
                    expected_sentence = str(eval_sentences[expected_sentence_idx]).strip().lower()
                    
                    print(f"Debug: Verifying last sentence...")
                    print(f"  CSV last sentence: '{last_csv_sentence[:60]}...'")
                    print(f"  Expected at idx {expected_sentence_idx}: '{expected_sentence[:60]}...'")
                    
                    if last_csv_sentence == expected_sentence:
                        print(f"Verified! Resuming from idx {resume_idx_candidate}")
                        return set(range(resume_idx_candidate))
                    else:
                        print("MISMATCH! Starting fresh (order changed?)")
                        return set()
                else:
                    print(f"Debug: File longer than evalsentences, starting fresh")
                    return set()
            else:
                print("Debug: No 'sentence' column found")
                return set()
                
        except Exception as e:
            print(f"Debug: CSV read failed: {e}, starting fresh")
            return set()

    # ==================== AFTER get_resume_idxs() ====================
    existing_done_idxs = get_resume_idxs(preds_path, eval_sentences)

    # Headers (already correct - sentence first, no idx)
    preds_header = ["idx", "sentence"] + CURRENT_FEATURES
    rats_header = ["idx", "sentence"] + CURRENT_FEATURES

    if not os.path.exists(preds_path):
        with open(preds_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(preds_header)
    if not os.path.exists(rats_path):
        with open(rats_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(rats_header)

    usable_ctx_count = 0
    used_two_turn_count = 0
    used_single_turn_count = 0

    start_time = time.time()
    print("###############################")
    print("start time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    print("###############################")

    
    # -------------------- GEMINI CACHING SETUP --------------------
    if args.backend == "gemini":
        print("Initializing Gemini Cache...")
        
        # 1. Generate the static system prompt using a dummy sentence
        # We use a dummy because build_messages requires an utterance, 
        # but we only want the system/feature block part.
        dummy_msgs, _ = build_messages(
            utterance="DUMMY",
            features=CURRENT_FEATURES,
            base_feature_block=BASE_FEATURE_BLOCK,
            instruction_type=args.instruction_type,
            include_examples_in_block=include_examples_in_block,
            dialect_legitimacy=args.dialect_legitimacy,
            self_verification=args.self_verification,
            require_rationales=require_rationales,
        )
        
        # Extract the "System" part (The massive rule block)
        system_content = next((m['content'] for m in dummy_msgs if m['role'] == 'system'), None)
        
        if system_content:
            # Create the cache on the server
            backend.create_cache(system_content, args.model)
        else:
            print("Warning: Could not isolate system prompt for caching.")
    # ==================== MAIN LOOP (idx is just loop counter) ====================
    for idx, sentence in enumerate(tqdm(eval_sentences, desc="Evaluating sentences")):
        usable = False
        arm_used = "no_context"
        context_included = False
        left = None
        right = None
        # Skip already-processed rows
        if idx in existing_done_idxs:
            continue
        
        left = None
        right = None
        
        # Get context from neighboring rows (uses idx as row position in golddf)
        if args.context:
            if idx > 0:
                left = golddf.loc[idx - 1, 'sentence']
            if idx < len(golddf) - 1:
                right = golddf.loc[idx + 1, 'sentence']
            usable = has_usable_context(left, right)
            if args.context and usable:
               usable_ctx_count += 1
        
        context_included = bool(args.context and usable)
        
        # Call model
        raw, arm_used = query_model(
            backend,
            enc,
            sentence,
            features=CURRENT_FEATURES,
            base_feature_block=BASE_FEATURE_BLOCK,
            instruction_type=args.instruction_type,
            include_examples_in_block=include_examples_in_block,
            use_context=args.context,
            left_context=left,
            right_context=right,
            context_mode=args.context_mode,
            dialect_legitimacy=args.dialect_legitimacy,
            self_verification=args.self_verification,
            require_rationales=require_rationales,
            dump_prompt=args.dump_prompt,
            dump_prompt_path=args.dump_prompt_path,
            dump_once_key=dumponcekey,
        )
        
        time.sleep(5) 
   
        if arm_used == "twoturn":
            used_two_turn_count += 1
        elif arm_used == "singleturn":
            used_single_turn_count += 1
        
        if not raw:
            parse_status = "EMPTYRESPONSE"
            missing_count = ""
            missing_keys_str = ""
            writemeta(idx, sentence, usable, arm_used, context_included, parse_status, missing_count, missing_keys_str)
            continue
        
        # Parse output
        try:
            vals, rats, missing = parse_output_json(raw, CURRENT_FEATURES)
            parse_status = "OK"
            missing_count = len(missing)
            missing_keys_str = ",".join(missing)
        except Exception:
            parse_status = "PARSEFAIL"
            missing_count = ""
            missing_keys_str = ""
            vals = {feat: None for feat in CURRENT_FEATURES}
            rats = {feat: "" for feat in CURRENT_FEATURES}
        
        writemeta(idx, sentence, usable, arm_used, context_included, parse_status, missing_count, missing_keys_str)
        
        # Write predictions (sentence first, NO idx)
        with open(preds_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([idx, sentence] + [vals.get(feat) for feat in CURRENT_FEATURES])
        
        # Write rationales (sentence first, NO idx)
        with open(rats_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([idx, sentence] + [rats.get(feat) for feat in CURRENT_FEATURES])

    print("\n=== CONTEXT USAGE SUMMARY ===")
    print(f"Sentences with usable context: {usable_ctx_count}")
    print(f"  - Single-turn context delivery: {used_single_turn_count}")
    print(f"  - Two-turn context delivery: {used_two_turn_count}")
    print_final_usage_summary()
    
    end_time = time.time()
    print("###############################")
    print("end time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    print("elapsed time:", time.strftime("%H:%M:%S", time.gmtime(end_time - start_time)))
    print("###############################")

if __name__ == "__main__":
    main()
