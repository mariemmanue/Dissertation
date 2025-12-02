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
     --file data/Test1.xlsx \
     --sheet Exp2 \
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
     --file data/Test1.xlsx \
     --sheet Exp1
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
# rule-first definitions with 1 pos + 1 Miss for hard ones
MASIS_FEATURE_BLOCK = """
1. zero-poss
   Decision rule (single semantic element: POSSESSION):
   Mark 1 if a possessive relationship is expressed WITHOUT an overt SAE possessive morpheme ('s) or standard possessive pronoun form,
   and this possessive meaning is clearly licensed within the same clause.
   Typical forms:
     • Noun–noun juxtaposition (dad car, mama house)
     • Bare or nonstandard possessive pronouns before a noun (they car, her brother kids).
   + Example (1): "That they dad boo." = 1 (nonstandard 'they' + 'dad boo' expressing possession)
   - Miss (0): "That's their dad's car." = 0 (all possessives fully marked in SAE)
   Ambiguity note:
   • If it is unclear whether two adjacent nouns form a possessive relationship or just a list/name (e.g., “school bus stop”), prefer 0 and mention ambiguity.

2. zero-copula
   Decision rule (single element: COPULAR / AUXILIARY BE DELETION):
   Mark 1 when a form of BE (is/are/was/were) that SAE requires is missing:
     (a) between a subject and a predicate (NP, AdjP, PP), OR
     (b) before a V-ing verb in a progressive construction,
   AND the material forms a single clause (not obviously subordinate or a list fragment).
   + Example (1): "She finna be strict." = 1 (missing 'is' between subject and V-ing verb 'finna')
   - Miss (0): "No problem." = 0 (formulaic fragment; no clear subject–predicate copula slot)
   Ambiguity note:
   • If the utterance can be read as a fragment or headline where SAE might also drop BE, prefer 0 and explain.

3. double-tense
   Decision rule (single element: DUPLICATED PAST MORPHOLOGY):
   Mark 1 when a single lexical verb shows duplicated overt past-tense morphology (usually repeated -ed) within one word.
   + Example (1): "She likeded me the best." = 1 (duplicated -ed)
   - Miss (0): "She liked me the best." = 0
   Ambiguity note:
   • Spelling variants that do not clearly reflect two past morphemes (e.g., typos) should be 0 unless duplication is obvious.

4. be-construction
   Decision rule (single element: UNINFLECTED BE WITH HABITUAL/GENERIC MEANING):
   Mark 1 when uninflected 'be' appears as a finite verb expressing habitual, iterative, or generalized action/state,
   not as an auxiliary of another tense or as an agreement error.
   + Example (1): "They be playing outside." = 1 (habitual activity)
   - Miss (0): "They is playing outside." = 0 (agreement error but not habitual 'be')
   Ambiguity note:
   • If 'be' could plausibly be a quoting frame, copula in a fixed phrase, or an auxiliary mis-agreement without clear habitual reading, prefer 0 and explain.

5. resultant-done
   Decision rule (single element: COMPLETIVE DONE):
   Mark 1 when preverbal 'done' directly precedes a verb phrase and marks completed aspect for that event. Adverbs or discourse markers may intervene.
   The following verb may be past-marked or bare, but the reading must be “already/completely finished.”
   + Example (1): "He done already ate it." = 1 (completed eating)
   - Miss (0): "They done it yesterday." = 0 (main verb 'done', not aspect marker)
   Ambiguity note:
   • If 'done' can be parsed as the main verb ‘did’ with no clear aspectual reading, prefer 0 and note ambiguity.

6. finna
   Decision rule (single element: IMMINENT/FUTURE FINNA):
   Mark 1 when 'finna' (or 'finta', 'fitna', etc.) functions as a preverbal marker meaning 'about to / fixing to', creating an imminent or near-future reading.
   + Example (1): "We finna eat." = 1 (imminent future)
   - Miss (0): "We gonna eat." = 0 (future, but not the AAE form 'finna')
   Ambiguity note:
   • Creative spellings close to 'gonna' or 'goin to' should NOT be marked unless they clearly instantiate the 'finna' paradigm.

7. come
   Decision rule (single element: STANCE 'COME' BEFORE VP):
   Mark 1 when 'come' is used as a preverbal stance/evaluative marker introducing a verb phrase or V-ing form describing someone's behavior, attitude, or speech,
   not as a simple motion verb.
   + Example (1): "He come talking mess again." = 1 (stance 'come' + V-ing)
   - Miss (0): "He came and talked." = 0 (motion + coordination, no stance meaning)
   Ambiguity note:
   • If 'come' can be interpreted purely as motion toward a place with no clear stance/attitude meaning, prefer 0.

8. double-modal
   Decision rule (single element: TWO MODALS IN ONE VERB CHAIN):
   Mark 1 when two modal-like elements appear in sequence in a single clause (e.g., 'might could', 'woulda could', 'used to could') and together scope over the following verb.
   + Example (1): "I might could go." = 1
   - Miss (0): "I might go." = 0 (single modal)
   Ambiguity note:
   • Sequences where the first item is clearly not a modal (e.g., main verb 'used') should be 0 unless it unambiguously patterns as a modal.

9. multiple-neg
   Decision rule (single element: MORE THAN ONE NEGATIVE FORM, ONE NEGATIVE MEANING):
   Mark 1 when two or more negative elements (negative auxiliaries, adverbs, pronouns, or determiners) appear within the same clause or phrase,
   but the overall meaning is a single logical negation.
   + Example (1): "I ain't never heard of that." = 1 (ain't + never)
   - Miss (0): "I ain't ever heard of that." = 0 (single negation)
   Notes:
   • This is the broad category covering both negative inversion (10) and negative concord (11).
   • If either 10 or 11 applies, this feature (multiple-neg) must also be 1.
   Ambiguity note:
   • If one negation belongs to a different clause (e.g., complement clause) and the sentence may have two separate negated propositions, prefer 0 and explain.

10. neg-inversion
    Decision rule (single element: NEGATIVE AUX BEFORE SUBJECT):
    Mark 1 when a negative auxiliary or marker appears before the subject in the clause, with the subject following it (a subtype of multiple negation).
    + Example (1): "Don’t nobody like how they actin’." = 1 (Don’t + subject 'nobody')
    – Miss (0): "Nobody don’t like how they actin’." = 0 (subject precedes negative auxiliary, no inversion)
    Notes:
    • Only mark in addition to multiple-neg when subject follows the negative auxiliary.
    Ambiguity note:
    • If word order is unclear due to disfluency, prefer 0 and note ambiguity.

11. n-inv-neg-concord
    Decision rule (single element: SUBJECT + VERB BOTH NEGATIVE, SUBJECT BEFORE NEGATIVE AUX)):
    Mark 1 when both the subject and the finite verb (or auxiliary) carry negative marking, but subject still precedes the verb (no inversion), forming one negative meaning.
    + Example (1): "Nobody don’t wanna see that." = 1 (negative subject + negative auxiliary)
    – Miss (0): "Nobody wanna see that." = 0 (subject negative, verb positive)
    Notes:
    • Only mark in addition to multiple-neg when both subject and verb are negative without inversion.
    Ambiguity note:
    • If one element is only pragmatically negative or unclear, prefer 0.

12. aint
    Decision rule (single element: GENERAL NEGATOR 'AIN'T'):
    Mark 1 when 'ain’t' is used as a general negative auxiliary for BE, HAVE, or DO, or as a general clausal negator, rather than as a lexical verb.
    + Example (1): "She ain't here." = 1 (negated copula)
    - Miss (0): "She isn't here." = 0 (not 'ain’t')
    Ambiguity note:
    • If the token could be 'ain’' or part of a different word and the segmentation is unclear, prefer 0.

13. zero-3sg-pres-s
    Decision rule (single element: MISSING -S ON 3SG PRESENT VERB):
    Mark 1 when a 3rd person singular subject (he, she, it, this/that NP, nobody, somebody, etc.) co-occurs with a bare verb or uninflected auxiliary (do, have, walk, go) in PRESENT-TENSE meaning,
    where SAE would require -s or does/has.
    Exclude non-agreeing 'is/was' forms (those are is-was-gen).
    + Example (1): "She walk to they house." = 1 (3sg subject + bare 'walk' with present meaning)
    - Miss (0): "They walk to their house." = 0 (plural subject = no 3sg requirement)
    Ambiguity note:
    • If the time reference may be generic/habitual but subject is not clearly 3sg, prefer 0 and explain.

14. is-was-gen
    Decision rule (single element: GENERALIZED IS/WAS WITH NON-STANDARD AGREEMENT):
    Mark 1 when 'is' or 'was' is used in a way that ignores SAE person/number agreement (e.g., with plural or 1st person subjects) in a finite clause,
    EXCEPT in existential 'it' constructions where SAE also allows is/was.
    + Example (1): "They was there." = 1 (plural subject + was)
    - Miss (0): "He was there." = 0 (SAE-agreeing)
    Notes:
    • Do not mark for existential/dummy 'it' constructions (“It was a fight,” “It’s people out here”), which are grammatical in SAE.
    Ambiguity note:
    • If 'was' may be part of a quoting frame or reported speech with unclear subject, prefer 0.

15. zero-pl-s
    Decision rule (single element: MISSING -S ON CLEAR PLURAL NOUN):
    Mark 1 when a noun that clearly has plural reference (from a quantifier, determiner, or context) surfaces without SAE plural -s, and the plural reading is local to the noun phrase.
    + Example (1): "She got them dog." = 1 (plural demonstrative 'them' + bare 'dog')
    - Miss (0): "A dogs." = 0 (article–noun mismatch, not the AAE plural pattern)
    Ambiguity note:
    • If plurality is only inferable from distant context and not clear in the NP itself, prefer 0.

16. double-object
    Decision rule (single element: TWO NP OBJECTS, NO PREPOSITION):
    Mark 1 when a verb is followed directly by two noun phrases (recipient + theme) in a single clause, with no preposition marking the recipient.
    + Example (1): "He gave him a lick." = 1 (verb + two NP objects)
    - Miss (0): "He gave it to her." = 0 (preposition 'to' introduces recipient)
    Notes:
    • Exclude clausal or wh-word complements (e.g., 'tell you what', 'show you how') where the second constituent is not a full NP.
    Ambiguity note:
    • If the second element might be a clause or small clause rather than an NP, prefer 0.

17a. wh-qu1  (WH-word + copula or DO deletion)
    Decision rule (single element: MISSING BE/DO IN WH-QUESTION):
    Mark 1 when a WH-question or WH-clause clearly requires a form of BE or DO in SAE, but that auxiliary is missing in the surface form.
    This includes:
      • Missing copula before a predicate or locative (Where she Ø at?)
      • Missing DO-support in WH-questions (What you Ø want?, Where you Ø go?)
    + Example (1): "Where she at?" = 1  (missing 'is')
    + Example (2): "What you want?" = 1  (missing 'do')
    - Miss (0): "Where is she?" = 0  (auxiliary present)
    - Miss (0): "What did you want?" = 0  (DO-support present)
    Notes:
    • Only mark wh-qu1 when the missing auxiliary is required for a WH-question in SAE.
    • Do not mark for simple topicalization or fragments that are not clearly questions.
    Ambiguity note:
    • If there is no clear question intonation or punctuation and the string might be a fragment, prefer 0.

17b. wh-qu2  (WH-word + non-standard inversion)
    Decision rule (single element: NON-SAE WH WORD ORDER):
    Mark 1 when a WH-question or WH-clause departs from SAE subject–auxiliary inversion patterns:
      • No inversion where SAE requires it (Where he is going? instead of Where is he going?)
      • Inversion inside embedded WH-clauses where SAE keeps declarative order
        (I asked him could he find her instead of I asked him if he could find her).
    + Example (1): "Where he is going?" = 1  (auxiliary follows subject in a main question)
    + Example (2): "I asked him could he find her." = 1  (inversion in embedded clause)
    - Miss (0): "Where is he going?" = 0  (standard WH inversion)
    - Miss (0): "I asked him if he could find her." = 0  (no embedded inversion)
    Notes:
    • Only mark wh-qu2 when WH-clause word order is non-standard relative to SAE.
    • If both wh-qu1 (aux deletion) and wh-qu2 (non-inversion) clearly apply, you may mark both.
    Ambiguity note:
    • If speech is heavily disfluent and apparent word order may be just a restart, prefer 0 and mention ambiguity.
"""

NEW_FEATURE_BLOCK = MASIS_FEATURE_BLOCK + """

18. existential-it
    Decision rule (single element: EXISTENTIAL 'IT' INSTEAD OF 'THERE'):
    Mark 1 when 'it' functions as an existential/dummy subject in a construction where SAE would normally use 'there' to introduce an existential,
    and the following predicate introduces new entities.
    + Example (1): "It’s people out here don’t care." = 1 (existential 'it' + people)
    - Miss (0): "It is raining out here." = 0 (weather 'it' is grammatical in SAE)
    Ambiguity note:
    • If 'it' can be read as a true referential pronoun with a clear antecedent, prefer 0.

19. demonstrative-them
    Decision rule (single element: 'THEM' AS DEMONSTRATIVE DETERMINER):
    Mark 1 when 'them' is used directly before a noun as a demonstrative determiner meaning 'those', not as an object pronoun.
    + Example (1): "Them shoes tight." = 1 (demonstrative determiner)
    - Miss (0): "I like them." = 0 (object pronoun)
    Ambiguity note:
    • If 'them' is separated from the noun or can be interpreted as a pronoun, prefer 0.

20. appositive-pleonastic-pronoun
    Decision rule (single element: REDUNDANT/RESUMPTIVE PRONOUN):
    Mark 1 when a subject or object NP is followed by a co-referential pronoun in the same clause, forming an appositive or pleonastic structure used for emphasis or clarity,
    not merely a disfluent restart.
    Fillers or pauses (e.g., 'uh', 'you know') may appear between NP and pronoun.
    + Example (1): "My dad, he told me it." = 1 (NP 'my dad' + resumptive 'he')
    - Miss (0): "My mama told me that." = 0 (no pronoun repetition)
    Ambiguity note:
    • If the structure could equally be a self-correction or restart with a new subject, and not clearly a redundant pronoun, prefer 0 and mention disfluency.

21. bin
    Decision rule (single element: BIN W/O 'HAVE'):
    Mark 1 when stressed BIN/BEEN (often capitalized in transcripts) appears without auxiliary 'have' and indicates that a state or action has been true for a long time
    (remote past continuing to present or at least long-established).
    + Example (1): "She BIN married." = 1 (long-standing state)
    - Miss (0): "She been married for two years." = 0 (unstressed, recent past; standard 'have been')
    Ambiguity note:
    • If stress is not marked and context does not clearly indicate remote/long-standing aspect, prefer 0.

22. verb-stem
    Decision rule (single element: BARE VERB WITH CLEAR PAST REFERENCE):
    Mark 1 when a bare (uninflected) verb form is used to express a clearly past event in the same clause, based on explicit temporal adverbs, surrounding context,
    or aspect markers (e.g., done), where SAE would require a past-tense form.
    + Example (1): "Yesterday he done walk to school." = 1 (bare 'walk' in a past context; also resultant-done = 1)
    - Miss (0): "He walk to school every day." = 0 (present habitual; possible zero-3sg-pres-s but not verb-stem past)
    Ambiguity note:
    • If there is no explicit evidence that the event is past (no time adverb, no clear past context), prefer 0 and avoid assuming past meaning.

23. past-tense-swap
    Decision rule (single element: NON-SAE TENSE FORM SUBSTITUTED IN SIMPLE PAST OR PAST PARTICIPLE POSITION):  
    Mark 1 when:
      • A past participle form is used as a simple past (e.g., seen, done, went as preterites), OR
      • A regularized past is used where an irregular past is expected,
    and the clause refers to a simple past event (not pluperfect).
    + Example (1): "I seen him yesterday." = 1 (past participle 'seen' used as preterite)
    – Miss (0): "I saw him yesterday." = 0 (standard preterite)
    Ambiguity note:
    • If the time reference is unclear and the form could belong to a perfect construction (with omitted auxiliary), prefer 0 and note ambiguity.

24. zero-rel-pronoun
    Decision rule (single element: MISSING SUBJECT RELATIVE PRONOUN):
    Mark 1 when a finite clause modifies a noun and functions as a subject relative, but there is NO overt relative pronoun ('who', 'that', 'which') in subject position:
      NP [Ø + finite clause].
    + Example (1): "There are many mothers [Ø don’t know their children]." = 1 (finite clause modifying 'mothers' without 'who')
    - Miss (0): "There are many mothers who don’t know their children." = 0 (overt 'who')
    Notes:
    • Exclude reduced relatives ('the guy wearing red'), appositives, or complement clauses not modifying the noun ('I know [that he left]').
    Ambiguity note:
    • If the clause could just as easily be a separate main clause rather than a modifier of the NP, prefer 0.

25. preterite-had
    Decision rule (single element: 'HAD' + VERB FOR SIMPLE PAST, NO PAST-BEFORE-PAST):
    Mark 1 when 'had' plus a past verb is used to express a simple past event (with no clear 'past-before-past' meaning),
    often with regularized/AAE-style past forms (had went, had ran), and there is no later past event anchoring a pluperfect reading.
    + Example (1): "The alarm next door had went off a few minutes ago." = 1 (simple past meaning; no later reference event)
    - Miss (0): "They had seen the movie before we arrived." = 0 (true pluperfect = past-before-past)
    Notes:
    • Accept overregularized forms (had went, had ran) as long as the temporal structure is simple past.
    Ambiguity note:
    • If there is a clear second past event that makes 'had' plausibly pluperfect, prefer 0 and treat as SAE pluperfect, not preterite-had.
"""


# Utility function definitions
def drop_features_column(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=['FEATURES', 'Source'], errors='ignore')

# def build_system_msg():
#     """System message emphasizing explicit syntactic analysis"""
#     return {
#         "role": "system",
#         "content": (
#             "You are a linguistic expert specializing in African American English (AAE). "
#             "You must analyze sentences using explicit syntactic analysis, NOT implicit inference.\n\n"
#             # 2. Dialect Validation: Explicitly state that African American English is a rule-governed linguistic system distinct from Standard American English. Instruct the model to analyze the input using AAVE's grammatical rules, rather than defaulting to SAE norms, to prevent misinterpretation of features as errors.
# # Retain and strengthen: "You are an experienced sociolinguist and expert annotator of African American English (AAE). AAE is a rule-governed, systematic language variety. You must analyze the input according to AAE's internal grammatical rules, not Standard American English (SAE) norms."
#             "CRITICAL REQUIREMENTS:\n"
#             "1. ALWAYS identify the main clause FIRST before making any feature decisions.\n"
#             "2. Identify the grammatical subject and verb of the main clause explicitly.\n"
#             "3. Base ALL feature decisions ONLY on the identified syntactic components.\n"
#             "4. Provide binary-form rationales tied to grammatical rules, NOT descriptive restatements.\n"
#             "5. Use metalinguistic justification referencing specific grammatical structures.\n"
#             "6. Suppress descriptive restatements of the sentence content.\n"
#             "7. For each feature, you must follow the structured format exactly.\n"
#             "8. Do NOT make inferences about implied structures—only analyze what is explicitly present.\n"
#             "9. Focus on clause-level justification, especially for negation, embedded clauses, and verb tense.\n"
        
# # You are a highly experienced sociolinguist and expert annotator of African American English (AAE). 
# # AAE is a rule-governed, systematic language variety. You must analyze the input according to AAE's 
# # internal grammatical rules, not Standard American English (SAE) norms. Your goal is to analyze the 
# # underlying GRAMMATICAL FORM of the input, focusing ONLY on syntax, morphology, and clause structure. 
# # Ignore phonetic or orthographic variations that do not affect the grammatical features, and ignore 
# # speaker identity, race, or social stereotypes.

        
        
        
        
#         ),
#     }

# def build_system_msg():
#     return {
#         "role": "system",
#         "content": (
#             "You are a linguist specializing in African American English (AAE). "
#             "Your job is to decide, for every sentence, whether EACH AAE feature from the list "
#             "is present (1) or absent (0), and to give a short grammatical rationale.\n\n"
#             "GENERAL RULES:\n"
#             "1. Always output ALL features, even if most are 0.\n"
#             "2. Use the predefined decision rule first before looking at examples for context.\n"
#             "3. If the sentence is a fragment or has disfluencies, only mark a feature if the target construction is explicitly realized (clear and not implied).\n"
#             "4. If the deciison is ambiguous, prefer precision over recall: output 0 and explain the ambiguity.\n"
#             "5. Use explicit temporal expressions (e.g., ‘yesterday’, ‘last week’) and context from surrounding verbs to clarify tense. If nearby verbs are also AAE features (e.g., verb-stem), compare their usage to SAE to avoid confusion.\n"
#             "6. Examine the broader context of the utterance, focusing on key segments and ignoring fillers. Pay attention to verb construction.\n"
#             "7. AAE features can co-occur (e.g., wh-question + zero copula). Mark all relevant features.\n"
#             "8. Focus on verb forms and tense markers to distinguish between tense-related features like is-was-gen, zero-3sg-pres-s, verb-stem, past-tense-swap, and preterite-had. Evaluate surrounding context to avoid misclassification.\n"
#             "9. Use context to identify dropped or implied subjects, unless ambiguity prevents clear identification.\n"
#         ),
#     }

# def build_system_response_instructions(utterance, features):
#     feature_list_str = ", ".join(f'"{f}"' for f in features)

#     return (
#         "Now analyze this utterance:\n"
#         f"UTTERANCE: {utterance}\n\n"
#         f"Return ONLY a single JSON object with EXACTLY these {len(features)} keys, in this order:\n"
#         f"[{feature_list_str}]\n\n"
#         "For each key, set:\n"
#         "- \"value\": 1 or 0\n"
#         "- \"rationale\": 1–2 sentences explaining the decision using the rule above.\n"
#         "Example output format (structure only):\n"
#         "{\n"
#         '  \"zero-poss\": {\"value\": 0, \"rationale\": \"...\"},\n'
#         '  \"zero-copula\": {\"value\": 1, \"rationale\": \"...\"},\n'
#         "  ...\n"
#         f'  \"{features[-1]}\": {{\"value\": 0, \"rationale\": \"...\"}}\n' 

#         # Add Example (Few-Shot/Rule Augmentation): 1. zero-poss (ZP): - RULE: A possessive meaning is expressed without an overt possessive marker 's or possessive pronoun form. - Example Span: Rolanda bed (for "Rolanda's bed")P
#         "}\n"
#         "Do not add fields. Do not change key names. Do not explain outside the JSON.\n"
#     )

def build_system_msg():
    """System message with principle-first, CoT, constraints, and dialect legitimacy."""
    return {
        "role": "system",
        "content": (
            "You are a highly experienced sociolinguist and expert annotator of African American English (AAE).\n"
            "AAE is a rule-governed, systematic language variety. You must analyze the input according to AAE's "
            "internal grammatical rules, not Standard American English (SAE) norms.\n\n"
            "Your task is to make strictly binary decisions (1 = present, 0 = absent) about a fixed list of AAE "
            "morphosyntactic features for a single target utterance.\n\n"
            "PROCEDURE (follow these steps, but keep explanations concise):\n"
            "1. CLAUSE & TENSE ANALYSIS (global):\n"
            "   - Identify the main clause of the TARGET UTTERANCE.\n"
            "   - Identify its grammatical subject(s), finite verb(s), and any auxiliaries.\n"
            "   - Identify tense/aspect markers (e.g., done, BIN, had) and overt temporal expressions (yesterday, last week).\n"
            "   - Identify the scope and pattern of negation and any embedded clauses.\n"
            "2. FEATURE-BY-FEATURE EVALUATION (single semantic element per feature):\n"
            "   - For each feature, first recall the decision rule and what semantic/grammatical element it targets "
            "     (e.g., possession, habitual aspect, multiple negation, WH word order, past vs present reference).\n"
            "   - Check whether the TARGET UTTERANCE satisfies ALL parts of that rule using only the visible syntax and morphology.\n"
            "   - Focus each decision on that single element (e.g., for verb-stem, focus only on past meaning with bare verb; "
            "     for appositive-pleonastic-pronoun, focus only on redundant pronouns).\n"
            "3. MULTIPLE INTERPRETATIONS:\n"
            "   - If more than one grammatical analysis is plausibly compatible with the utterance, briefly consider the strongest "
            "     alternative.\n"
            "   - Choose the decision (1 or 0) that is most consistent with the feature's decision rule.\n"
            "   - If evidence is genuinely insufficient to decide, prefer precision over recall: output 0 and state that the "
            "     structure is ambiguous.\n"
            "4. SELF-VERIFICATION (internal reliability check):\n"
            "   - Before finalizing each feature's rationale, quickly check that:\n"
            "       * Your rationale cites only grammatical facts from the sentence (word order, morphology, clause boundaries).\n"
            "       * You did NOT rely on world knowledge, stereotypes, or assumptions about what the speaker 'meant'.\n"
            "       * You did NOT treat an AAE pattern as an 'error'; you described it as an AAE construction compared to SAE.\n\n"
            "EXPLICIT EVALUATION CONSTRAINTS:\n"
            "- Analyze ONLY syntax, morphology, and clause structure.\n"
            "- Treat fillers and discourse markers (e.g., 'like', 'you know', 'I mean', 'uh', 'um') as non-grammatical material "
            "  unless they clearly function as verbs, complementizers, or clause heads. Do NOT base feature decisions purely on fillers.\n"
            "- For tense-related features (verb-stem, past-tense-swap, preterite-had, is-was-gen, zero-3sg-pres-s):\n"
            "   * First determine the intended reference time (past vs present vs habitual) using explicit time adverbs, aspect markers, "
            "     and discourse context.\n"
            "   * Then compare the verb form to what SAE would require in that same context.\n"
            "   * Only mark the feature as 1 when the mismatch matches the defined AAE pattern.\n"
            "- For 'BIN', 'done', and 'had', distinguish long-standing vs recent or pluperfect readings using context; mark the feature only when "
            "  the specialized AAE reading is clearly licensed by the utterance.\n"
            "- Do NOT infer missing words or tense from plausibility in the real world; use only textual evidence.\n\n"
            "- Describe structures as 'AAE pattern X' versus 'SAE pattern Y' in your reasoning when useful.\n"
            "OUTPUT CONSTRAINTS:\n"
            "- You MUST output decisions for EVERY feature key provided.\n"
            "- Each feature's 'value' MUST be exactly 1 or 0 (no other numbers or strings).\n"
            "- Each 'rationale' MUST be 1–2 sentences, focusing on the single semantic/grammatical element relevant to that feature "
            "  (e.g., possession for zero-poss, negation structure for multiple-neg, tense/aspect for verb-stem, WH word order for wh-qu2).\n"
            "- Do NOT talk about multiple unrelated features in one rationale. If you considered an alternative analysis, mention it "
            "  briefly inside that same rationale (e.g., 'Alt analysis: ... but rejected because ...').\n"
        ),
    }

def build_system_response_instructions(utterance, features):
    feature_list_str = ", ".join(f'"{f}"' for f in features)

    return (
        "Now analyze this utterance strictly according to the rules above:\n"
        f"UTTERANCE: {utterance}\n\n"
        "TASK:\n"
        "- For EACH feature in the list below, decide whether it is present (1) or absent (0) IN THIS UTTERANCE.\n"
        "- Treat each feature as a separate, small, task-grounded question about ONE semantic/grammatical element "
        "(e.g., possession, habitual aspect, negation configuration, WH word order, tense/aspect of the main verb).\n"
        "- Apply the procedure before making each binary decision.\n\n"
        f"FEATURE KEYS (in required order):\n[{feature_list_str}]\n\n"
        "OUTPUT FORMAT (STRICT):\n"
        "- Return ONLY a single JSON object.\n"
        "- It MUST contain EXACTLY these keys, in this order.\n"
        "- For each key, you MUST provide an object with:\n"
        "    'value': 1 or 0 (integer, not a string)\n"
        "    'rationale': 1–2 sentences of grammatical reasoning.\n"
        "- The rationale SHOULD:\n"
        "    * Point to the specific grammatical evidence (e.g., bare verb with past adverb, duplicated tense, WH + missing copula).\n"
        "    * If you considered an alternative interpretation, mention it briefly (e.g., 'Alt analysis: ... but the rule requires ...').\n"
        "    * Avoid paraphrasing the sentence content or adding world knowledge; stay on syntax/morphology.\n\n"
        "Example output structure (for illustration only):\n"
        "{\n"
        '  \"zero-poss\": {\"value\": 0, \"rationale\": \"No bare possessive noun–noun sequence; possessive is fully marked in SAE form.\"},\n'
        '  \"zero-copula\": {\"value\": 1, \"rationale\": \"Subject is followed by predicate adjective with no overt form of BE in a single clause.\"},\n'
        "  ...\n"
        f'  \"{features[-1]}\": {{\"value\": 0, \"rationale\": \"Word order in the WH-clause matches SAE, so this WH feature does not apply.\"}}\n'
        "}\n\n"
        "Do NOT add extra fields.\n"
        "Do NOT change key names.\n"
        "Do NOT include any explanation outside of this single top-level JSON object.\n"
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
                print(f"Error on sentence: {sentence[:40]}... = {e}")
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
                print(f"Error on sentence: {sentence[:40]}... = {e}")
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