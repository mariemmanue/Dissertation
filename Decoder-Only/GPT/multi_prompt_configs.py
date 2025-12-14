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

# ICL commands
"""
"""

OPENAI_MODEL_NAME = "gpt-5"
# tokenizer for logging (will be overridden by main)
enc = tiktoken.encoding_for_model("gpt-5")

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
    "bin", "verb-stem", "past-tense-swap", "zero-rel-pronoun", "preterite-had"
]

# -------------------- FEATURE BLOCKS (WITH EXAMPLES) --------------------
# NOTE: For zero-shot and zero-shot CoT, we strip + Example / - Miss lines programmatically,
# so we can keep the rich versions here.

MASIS_FEATURE_BLOCK = """
1. zero-poss
   Decision rule (single semantic element: POSSESSION):
   Mark 1 if a possessive relationship is expressed WITHOUT an overt SAE possessive morpheme ('s) or standard possessive pronoun form,
   and this possessive meaning is clearly licensed within the same clause.
   Typical forms:
     • Noun–noun juxtaposition (dad car, mama house)
     • Bare or nonstandard possessive pronouns before a noun (they car, her brother kids).
   + Example (label = 1): "That they dad boo." (nonstandard 'they' + 'dad boo' expressing possession)
   - Miss (label = 0): "That's their dad's car." (all possessives fully marked in SAE) 
   Ambiguity note:
   • If it is unclear whether two adjacent nouns form a possessive relationship or just a list/name (e.g., “school bus stop”), prefer 0 and mention ambiguity.

2. zero-copula
   Decision rule (single element: COPULAR / AUXILIARY BE DELETION):
   Mark 1 when a form of BE (is/are/was/were) that SAE requires is missing:
     (a) between a subject and a predicate (NP, AdjP, PP), OR
     (b) before a V-ing verb in a progressive construction,
   AND the material forms a single clause (not obviously subordinate or a list fragment).
   + Example (label = 1): "She finna eat." (missing 'is' between subject and V-ing (finna = fixing to))
   - Miss (label = 0): "No problem." (formulaic fragment; no clear subject–predicate copula slot)
   Ambiguity note:
   • If the utterance can be read as a fragment or headline where SAE might also drop BE, prefer 0 and explain.

3. double-tense
   Decision rule (single element: DUPLICATED PAST MORPHOLOGY):
   Mark 1 when a single lexical verb shows duplicated overt past-tense morphology (usually repeated -ed) within one word.
   + Example (label = 1): "She likeded me the best." (duplicated -ed)
   - Miss (label = 0): "She liked me the best."
   Ambiguity note:
   • Spelling variants that do not clearly reflect two past morphemes (e.g., typos) should be 0 unless duplication is obvious.

4. be-construction
   Decision rule (single element: UNINFLECTED BE WITH HABITUAL/GENERIC MEANING):
   Mark 1 when uninflected 'be' appears as a finite verb expressing habitual, iterative, or generalized action/state,
   not as an auxiliary of another tense or as an agreement error.
   + Example (label = 1): "They be playing outside." (habitual activity)
   - Miss (label = 0): "They is playing outside." (is/was generalization, not habitual 'be')
   Ambiguity note:
   • If 'be' could plausibly be a quoting frame, copula in a fixed phrase, or an auxiliary mis-agreement without clear habitual reading, prefer 0 and explain.

5. resultant-done
   Decision rule (single element: COMPLETIVE DONE):
   Mark 1 when preverbal 'done' directly precedes a verb phrase and marks completed aspect for that event. Adverbs or discourse markers may intervene.
   The following verb may be past-marked or bare, but the reading must be “already/completely finished.”
   + Example (label = 1): "He done already ate it." (completed eating)
   - Miss (label = 0): "They done it yesterday." (main verb 'done', not aspect marker)
   Ambiguity note:
   • If 'done' can be parsed as the main verb ‘did’ with no clear aspectual reading, prefer 0 and note ambiguity.

6. finna
   Decision rule (single element: IMMINENT/FUTURE FINNA):
   Mark 1 when 'finna' (or 'finta', 'fitna', etc.) functions as a preverbal marker meaning 'about to / fixing to', creating an imminent or near-future reading.
   + Example (label = 1): "We is finna eat." (imminent future)

7. come
   Decision rule (single element: STANCE 'COME' BEFORE VP):
   Mark 1 when 'come' is used as a preverbal stance/evaluative marker introducing a verb phrase or V-ing form describing someone's behavior, attitude, or speech,
   not as a simple motion verb.
   + Example (label = 1): "He come talking mess again." (stance 'come' + V-ing)
   - Miss (label = 0): "He come and talked." (motion + coordination)
   Ambiguity note:
   • If 'come' can be interpreted purely as motion toward a place with no clear stance/attitude meaning, prefer 0.

8. double-modal
   Decision rule (single element: TWO MODALS IN ONE VERB CHAIN):
   Mark 1 when two modal-like elements appear in sequence in a single clause (e.g., 'might could', 'woulda could', 'used to could') and together scope over the following verb.
   + Example (label = 1): "I might could go."
   - Miss (label = 0): "I might go." (single modal)

9. multiple-neg
   Decision rule (single element: MORE THAN ONE NEGATIVE FORM, ONE NEGATIVE MEANING):
   Mark 1 when two or more negative elements (negative auxiliaries, adverbs, pronouns, or determiners) appear within the same clause or phrase,
   but the overall meaning is a single logical negation.
   + Example (label = 1): "I ain't never heard of that." (ain't + never)
   - Miss (label = 0): "I ain't ever heard of that." (single negation)
   Notes:
   • This is the broad category covering both negative inversion (10) and negative concord (11).
   • If either 10 or 11 applies, this feature (multiple-neg) must also be 1.
   Ambiguity note:
   • If one negation belongs to a different clause (e.g., complement clause) and the sentence may have two separate negated propositions, prefer 0 and explain.

10. neg-inversion
    Decision rule (single element: NEGATIVE AUX BEFORE SUBJECT):
    Mark 1 when a negative auxiliary or marker appears before the subject in the clause, with the subject following it (a subtype of multiple negation).
    + Example (label = 1): "Don’t nobody like how they actin’." (Don’t + subject 'nobody')
    – Miss (label = 0): "Nobody don’t like how they actin’." (subject precedes negative auxiliary, no inversion)
    Notes:
    • Only mark in addition to multiple-neg when subject follows the negative auxiliary, at the beginning of a sentence or clause.
    Ambiguity note:
    • If word order is unclear due to disfluency, prefer 0 and note ambiguity.

11. n-inv-neg-concord
    Decision rule (single element: SUBJECT + VERB BOTH NEGATIVE, SUBJECT BEFORE NEGATIVE AUX)):
    Mark 1 when both the subject and the finite verb (or auxiliary) carry negative marking, but subject still precedes the verb (no inversion), forming one negative meaning.
    + Example (label = 1): "Nobody don’t wanna see that." (negative subject + negative auxiliary)
    – Miss (label = 0): "Nobody wanna see that." (subject negative, verb positive)
    Notes:
    • Only mark in addition to multiple-neg when both subject and verb are negative without inversion, at the beginning of a sentence or clause.
    Ambiguity note:
    • If one element is only pragmatically negative or unclear, prefer 0.

12. aint
    Decision rule (single element: GENERAL NEGATOR 'AIN'T'):
    Mark 1 when 'ain’t' is used as a general negative auxiliary for BE, HAVE, or DO, or as a general clausal negator, rather than as a lexical verb.
    + Example (1): "She ain't here." (negated copula)
    - Miss (0): "She isn't here." (not 'ain’t')

13. zero-3sg-pres-s
    Decision rule (single element: MISSING -S ON 3SG PRESENT VERB):
    Mark 1 when a 3rd person singular subject (he, she, it, this/that NP, nobody, somebody, etc.) co-occurs with a bare verb or uninflected auxiliary (do, have, walk, go) in PRESENT-TENSE meaning,
    where SAE would require -s or does/has.
    Exclude non-agreeing 'is/was' forms (those are is-was-gen).
    + Example (label = 1): "She walk to they house." (3sg subject + bare 'walk' with present meaning)
    - Miss (label = 0): "They walk to their house." (plural subject = no 3sg requirement)
    Ambiguity note:
    • If the time reference is in past tense, prefer 0 and explain. 

14. is-was-gen
    Decision rule (single element: GENERALIZED IS/WAS WITH NON-STANDARD AGREEMENT):
    Mark 1 when 'is' or 'was' is used in a way that ignores SAE person/number agreement (e.g., with plural or 1st person subjects) in a finite clause,
    EXCEPT in existential 'it' constructions where SAE also allows is/was.
    + Example (label = 1): "They was there." (plural subject + was)
    - Miss (label = 0): "He was there." (SAE-agreeing)
    Notes:
    • Do not mark for existential/dummy 'it' constructions (“It was a fight,” “It’s people out here”), which are grammatical in SAE.
    Ambiguity note:
    • If 'was' may be part of a quoting frame or reported speech with unclear subject, prefer 0.

15. zero-pl-s
    Decision rule (single element: MISSING -S ON CLEAR PLURAL NOUN):
    Mark 1 when a noun that clearly has plural reference (from a quantifier, determiner, or context) surfaces without SAE plural -s, and the plural reading is local to the noun phrase.
    + Example (label = 1): "She got them dog." (plural demonstrative 'them' + bare 'dog')
    - Miss (label = 0): "A dogs." (article–noun mismatch, not the AAE plural pattern)
    Ambiguity note:
    • If plurality is only inferable from distant context and not clear in the NP itself, prefer 0.

16. double-object
    Decision rule (single element: TWO NP OBJECTS, NO PREPOSITION):
    Mark 1 when a verb is followed directly by two noun phrases (recipient + theme) in a single clause, with no preposition marking the recipient.
    + Example (label = 1): "He gave him a lick." (verb + two NP objects)
    - Miss (label = 0): "He gave it to her." (preposition 'to' introduces recipient)
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
    + Example (label = 1): "Where she at?"  (missing 'is')
    + Example (label = 1): "What you want?"  (missing 'do')
    - Miss (label = 0): "Where is she?"  (auxiliary present)
    - Miss (label = 0): "What did you want?"  (DO-support present)
    Notes:
    • Only mark wh-qu1 AND zero-copula when the missing auxiliary is required for a WH-question in SAE.
    • Do not mark for simple topicalization or fragments that are not clearly questions.
    Ambiguity note:
    • If there is no clear question intonation or punctuation and the string might be a fragment, prefer 0.

17b. wh-qu2  (WH-word + non-standard inversion)
    Decision rule (single element: NON-SAE WH WORD ORDER):
    Mark 1 when a WH-question or WH-clause departs from SAE subject–auxiliary inversion patterns:
      • No inversion where SAE requires it
      • Inversion inside embedded WH-clauses where SAE keeps declarative order
    + Example (label = 1): "Where he is going?"  (auxiliary follows subject in a main question)
    + Example (label = 1): "I asked him could he find her." (inversion in embedded clause)
    - Miss (label = 0): "I asked him if he could find her."  (no embedded inversion)
    Notes:
    • Only mark wh-qu2 when WH-clause word order is non-standard relative to SAE.
    Ambiguity note:
    • If speech is heavily disfluent and apparent word order may be just a restart, prefer 0 and mention ambiguity.
"""

NEW_FEATURE_BLOCK = MASIS_FEATURE_BLOCK + """

18. existential-it
    Decision rule (single element: EXISTENTIAL 'IT' INSTEAD OF 'THERE'):
    Mark 1 when 'it' functions as an existential/dummy subject in a construction where SAE would normally use 'there' to introduce an existential,
    and the following predicate introduces new entities.
    + Example (label = 1): "It’s people out here don’t care." (existential 'it' + people)
    - Miss (label = 0): "It is raining out here." (weather 'it' is grammatical in SAE)
    Ambiguity note:
    • If 'it' can be read as a true referential pronoun with a clear antecedent, prefer 0.

19. demonstrative-them
    Decision rule (single element: 'THEM' AS DEMONSTRATIVE DETERMINER):
    Mark 1 when 'them' is used directly before a noun as a demonstrative determiner meaning 'those', not as an object pronoun.
    + Example (label = 1): "Them shoes tight." (demonstrative determiner)
    - Miss (label = 0): "I like them." (object pronoun)
    Ambiguity note:
    • If 'them' is separated from the noun or can be interpreted as a pronoun, prefer 0.

20. appositive-pleonastic-pronoun
    Decision rule (single element: REDUNDANT/RESUMPTIVE PRONOUN):
    Mark 1 when a subject or object NP is followed by a co-referential pronoun in the same clause, forming an appositive or pleonastic structure used for emphasis or clarity,
    not merely a disfluent restart.
    Fillers or pauses (e.g., 'uh', 'you know') may appear between NP and pronoun.
    + Example (label = 1): "My dad, he told me it." (NP 'my dad' + resumptive 'he')
    - Miss (label = 0): "My mama told me that." (no pronoun repetition)
    Ambiguity note:
    • If the structure could equally be a self-correction or restart with a new subject, and not clearly a redundant pronoun, prefer 0 and mention disfluency.

21. bin
    Decision rule (single element: BIN W/O 'HAVE'):
    Mark 1 when stressed BIN/BEEN (often capitalized in transcripts) appears without auxiliary 'have' and indicates that a state or action has been true for a long time
    (remote past continuing to present or at least long-established).
    + Example (label = 1): "She BIN married." = 1 (long-standing state)
    - Miss (label = 0): "She been married for two years." (unstressed, recent past; standard 'have been')

22. verb-stem
    Decision rule (single element: BARE VERB WITH CLEAR PAST REFERENCE):
    Mark 1 when a bare (uninflected) verb form is used to express a clearly past event in the same clause, based on explicit temporal adverbs, surrounding context,
    or aspect markers (e.g., done), where SAE would require a past-tense form.
    + Example (label = 1): "Yesterday he done walk to school." (bare 'walk' in a past context; also resultant-done = 1)
    - Miss (label = 0): "He walk to school every day." (present habitual; possible zero-3sg-pres-s but not verb-stem past)
    Ambiguity note:
    • If there is no explicit evidence that the event is past (no time adverb, no clear past context), prefer 0 and avoid assuming past meaning.

23. past-tense-swap
    Decision rule (single element: NON-SAE TENSE FORM SUBSTITUTED IN SIMPLE PAST OR PAST PARTICIPLE POSITION):  
    Mark 1 when:
      • A past participle form is used as a simple past (e.g., seen, done, went as preterites), OR
      • A regularized past is used where an irregular past is expected, and the clause refers to a simple past event.
    + Example (label = 1): "I seen him yesterday." (past participle 'seen' used as preterite)
    – Miss (label = 0): "I saw him yesterday." (standard preterite)
    Ambiguity note:
    • If the time reference is unclear and the form could belong to a perfect construction (with omitted auxiliary), prefer 0 and note ambiguity.

24. zero-rel-pronoun
    Decision rule (single element: MISSING SUBJECT RELATIVE PRONOUN):
    Mark 1 when a finite clause modifies a noun and functions as a subject relative, but there is NO overt relative pronoun ('who', 'that', 'which') in subject position.
    + Example (label = 1): "There are many mothers [Ø don’t know their children]." (finite clause modifying 'mothers' without 'who')
    - Miss (label = 0): "I think he left." (that-deletion in a complement clause)
    Notes:
    • Exclude reduced relatives ('the guy wearing red'), appositives, or complement clauses not modifying the noun ('I know [that he left]').
    Ambiguity note:
    • If the clause could just as easily be a separate main clause rather than a modifier of the NP, prefer 0.

25. preterite-had
    Decision rule (single element: 'HAD' + VERB FOR SIMPLE PAST, NO PAST-BEFORE-PAST):
    Mark 1 when 'had' plus a past verb is used to express a simple past event (with no clear 'past-before-past' meaning),
    often with regularized/AAE-style past forms (had went, had ran), and there is no later past event anchoring a pluperfect reading.
    + Example (label = 1): "The alarm next door had went off a few minutes ago." (simple past meaning; no later reference event)
    - Miss (label = 0): "They had seen the movie before we arrived." (true pluperfect = past-before-past)
    Notes:
    • Accept overregularized forms (had went, had ran) as long as the temporal structure is simple past.
    Ambiguity note:
    • If there is a clear second past event that makes 'had' plausibly pluperfect, prefer 0 and treat as SAE pluperfect, not preterite-had.
"""

# -------------------- ICL FEW-SHOT EXAMPLES BLOCK --------------------

ICL_EXAMPLES_BLOCK = """
FEW-SHOT TRAINING EXAMPLES (for demonstration only; NOT the target utterance).

Example 1:
SENTENCE: "And my cousin place turn into a whole cookout soon as it get warm, and when you step outside it's people dancing out on the sidewalk."
ANNOTATED LABELS (subset):
- zero-3sg-pres-s: {
    "value": 1,
    "rationale": "The clause 'turn into a whole cookout' has a recoverable 3sg subject from the local discourse ('my cousin place' / 'it'). Interpreting it as '[it] turn into a whole cookout', the verb 'turn' has a present/habitual reading with a 3sg subject. SAE would require 'turns', so this counts as a missing -s with an understood subject."
}
- existential-it: {
    "value": 1,
    "rationale": "In 'it's people dancing out on the sidewalk', 'it' functions as a dummy subject introducing new entities ('people'), patterning like existential 'it' (similar to 'there are people dancing') rather than a referential or weather 'it'."
}
- multiple-neg: {
    "value": 0,
    "rationale": "There are no negative elements in this utterance; there is no configuration with two or more negatives contributing to a single logical negation."
}
- zero-poss: {
    "value": 1,
    "rationale": "In 'my cousin place,' the noun 'place' expresses a possessive relationship to 'my cousin' without an overt SAE possessive marker ('my cousin's place'). The two nouns appear in direct sequence, forming the AAE-style zero-possessive construction. The meaning is clearly possessive rather than appositive or a compound noun."
}

Example 2:
SENTENCE: "He threwed him a quick punch and a hard knock to the ribs, then spin around and walk straight out the room like it was nothing."
ANNOTATED LABELS (subset):
- double-object: {
    "value": 1,
    "rationale": "In 'threwed him a quick jab and a hard hook', the verb 'threwed' is followed by the indirect object 'him' and then a direct object NP ('a quick jab and a hard hook') with no preposition introducing the recipient. This is a verb + two NP objects pattern, characteristic of the double-object construction."
}
- verb-stem-past: {
    "value": 1,
    "rationale": "The sentence describes a single past-time event established by the past-tense context ('threwed him...'). The coordinated verbs 'spin' and 'walk' appear in bare-stem form instead of the SAE narrative-past forms ('spun', 'walked'). Because the temporal reference has already been set to the past, the bare verbs function as past-time predicates, matching the AAE bare-stem past pattern."
}
- past-tense-swap: {
    "value": 1,
    "rationale": "'threwed' is an overregularized past form in place of the SAE irregular 'threw'. This matches the criterion for past-tense-swap, where a regularized or participle-like form is used as the simple past."
}

Example 3:
SENTENCE: "He ain't never seen nothing move that fast before, then a week later he just seen it happen again right in front of him."
ANNOTATED LABELS (subset):
- multiple-neg: {
    "value": 1,
    "rationale": "In 'ain't never seen nothing', 'ain't', 'never', and 'nothing' are all negative elements within the same clause, but they jointly express a single logical negation ('he has never seen anything like that'). This is the hallmark of multiple negation / negative concord."
}
- aint: {
    "value": 1,
    "rationale": "'ain't' here functions as a general negative auxiliary, standing in for 'hasn't'/'has not' with the participle 'seen' ('he hasn't ever seen anything'), rather than as a lexical main verb."
}
- past-tense-swap: {
    "value": 1,
    "rationale": "In 'he just seen it happen again', 'seen' (a participle form) is used as a simple past where SAE would require 'saw'. The time reference ('a week later' + event already completed) makes this a simple past, so this is a nonstandard substitution of a participle form in preterite position."
}
- verb-stem: {
    "value": 0,
    "rationale": "Neither 'seen' nor 'happen' is functioning as a bare-stem past: 'seen' is a participle used as simple past (captured by past-tense-swap), and 'happen' appears under 'seen it happen' without independent past marking. There is no clear case of a bare verb form directly realizing a past-time finite verb slot."
}

Use these as patterns for how to connect specific grammatical evidence to binary feature decisions.
Do NOT reuse these sentences when analyzing the new target utterance.
"""



# -------------------- UTILITIES --------------------

def drop_features_column(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=['FEATURES', 'Source'], errors='ignore')


def strip_examples_from_block(block: str) -> str:
    """
    Remove '+ Example' and '-/– Miss' lines from a feature block to create a rules-only version.
    This is used for zero-shot / zero-shot CoT conditions so that the feature block does not
    itself contain labeled examples.
    """
    stripped_lines = []
    for line in block.splitlines():
        s = line.lstrip()
        if s.startswith("+ Example"):
            continue
        if s.startswith("- Miss") or s.startswith("– Miss"):
            continue
        stripped_lines.append(line)
    return "\n".join(stripped_lines)


def build_system_msg(
    instruction_type: str,
    dialect_legitimacy: bool,
    self_verification: bool,
) -> dict:
    """
    Build the system message.

    - instruction_type: controls CoT vs non-CoT wording.
    - dialect_legitimacy: whether to explicitly frame AAE as rule-governed & not 'errors'.
    - self_verification: whether to include the explicit self-check step.
    """

    if dialect_legitimacy:
        intro = (
            "You are a highly experienced sociolinguist and expert annotator of African American English (AAE).\n"
            "AAE is a rule-governed, systematic language variety. You must analyze the input according to AAE's "
            "internal grammatical rules, not Standard American English (SAE) norms.\n"
            "Do NOT treat AAE constructions as 'incorrect' or 'broken' English; your job is to identify and justify "
            "AAE patterns relative to SAE, not to correct them.\n"
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
        "   - If more than one grammatical analysis is genuinely possible based on the surface structure, briefly acknowledge the\n"
        "     strongest alternative.\n"
        "   - Then choose the label (1 or 0) that best satisfies the feature’s *explicit* decision rule.\n"
        "   - If the utterance does not provide enough grammatical evidence to apply the rule, prefer precision over recall:\n"
        "       output 0 and explain that the structure is grammatically underspecified.\n"
        "   - Do NOT treat disfluencies, missing subjects, or casual phrasing as ambiguity unless they obscure the syntactic\n"
        "     environment relevant to the feature.\n\n"
        )

    if self_verification:
        base_content += (
            "4. SELF-VERIFICATION (FINAL CHECK BEFORE OUTPUT):\n"
            "   - For each feature, confirm that:\n"
            "       * Your reasoning uses ONLY grammatical facts in the utterance—word order, morphology, clause structure, tense/aspect.\n"
            "       * You are NOT filling in missing material with world knowledge, assumptions about intent, or reconstructed subjects.\n"
            "       * You are NOT labeling a structure as an AAE feature merely because it differs from SAE or sounds informal.\n"
            "         A feature must match its defined AAE grammatical pattern.\n"
            "       * Your rationale explains *why* the rule either applies or does not apply, even if the label is 0.\n\n"
        )
    else:
        base_content += "\n"

    base_content += (
        "EXPLICIT EVALUATION CONSTRAINTS:\n"
        "- Analyze ONLY syntax, morphology, and clause structure.\n"
        "- Informal, conversational, or slang (including missing subjects, contractions, or casual phrasing) "
        "  does NOT by itself imply an AAE feature. Only mark a feature when the specific AAE decision rule is satisfied.\n"
        "- SUBJECT DROPS IN SPONTANEOUS SPEECH:\n"
        "   - In conversational speech, speakers often omit subjects when they are pragmatically given.\n"
        "   - If a clause has a clearly recoverable subject from the SAME utterance or the immediately preceding clause,you may treat that subject as syntactically present when relevant.\n"
        "   - In these cases, compare the verb form to what SAE would require with that understood subject.\n"
        "   - If no specific subject is clearly recoverable from the local context (the clause could be a fragment), prefer 0\n"
        "     and briefly note that the structure is too fragmentary or ambiguous for a reliable feature decision.\n"
        "- For tense-related features (verb-stem, past-tense-swap, preterite-had, is-was-gen, zero-3sg-pres-s):\n"
        "   - First determine the intended reference time (past vs present vs habitual) using explicit time adverbs, aspect markers,\n"
        "     and LOCAL discourse context (earlier/later clauses in the same utterance or the provided context sentences).\n"
        "   - Then compare the verb form to what SAE would require in that same context.\n"
        "   - Only mark the feature as 1 when the mismatch fits the defined AAE pattern.\n"
        "- Do NOT infer missing words, tense, or subjects purely from real-world plausibility or stereotypes about what speakers\n"
        "  'probably meant.' Use only the textual and local discourse evidence that is actually present in the given context.\n\n"
    )


    if "cot" in instruction_type:
        base_content += (
            "ADDITIONAL REASONING REQUIREMENT (CHAIN-OF-THOUGHT STYLE):\n"
            "- Before filling the JSON, internally reason step by step about clause structure, tense/aspect, negation, and the "
            "  single semantic element for each feature.\n"
            "- Do NOT output those intermediate steps separately; only include the final binary values (and rationales, if requested) in the JSON.\n\n"
        )

    return {"role": "system", "content": base_content}


def build_system_response_instructions(
    utterance: str,
    features: list[str],
    instruction_type: str,
    require_rationales: bool,
) -> str:
    feature_list_str = ", ".join(f'"{f}"' for f in features)

    extra_cot_line = ""
    if "cot" in instruction_type:
        extra_cot_line = (
            "- Internally, you may reason step by step before deciding each feature, but do NOT print the steps.\n"
        )

    if require_rationales:
        output_format = (
            "OUTPUT FORMAT (STRICT):\n"
            "- Return ONLY a single JSON object.\n"
            "- It MUST contain EXACTLY these keys, in this order.\n"
            "- For each key, you MUST provide an object with:\n"
            "    'value': 1 or 0 (integer, not a string)\n"
            "    'rationale': 1–2 sentences of grammatical reasoning.\n"
            "- The rationale SHOULD:\n"
            "    - Point to the specific grammatical evidence (e.g., bare verb with past adverb, duplicated tense, WH + missing copula).\n"
            "    - If you considered an alternative interpretation, mention it briefly (e.g., 'Alt analysis: ... but the rule requires ...').\n"
            "    - Avoid paraphrasing the sentence content or adding world knowledge; stay on syntax/morphology.\n\n"
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
    else:
        # LABEL-ONLY CONDITION: flat JSON from feature -> 0/1, no rationale requests.
        output_format = (
            "OUTPUT FORMAT (STRICT, LABEL-ONLY):\n"
            "- Return ONLY a single JSON object.\n"
            "- It MUST contain EXACTLY these keys, in this order.\n"
            "- For each key, the value MUST be exactly 1 or 0 (integer, not a string).\n"
            "- Do NOT include nested objects, rationales, or any extra fields.\n\n"
            "Example output structure (for illustration only):\n"
            "{\n"
            '  \"zero-poss\": 0,\n'
            '  \"zero-copula\": 1,\n'
            "  ...\n"
            f'  \"{features[-1]}\": 0\n'
            "}\n\n"
            "Do NOT add extra fields.\n"
            "Do NOT change key names.\n"
            "Do NOT include any explanation outside of this single top-level JSON object.\n"
        )

    return (
        "Now analyze this utterance strictly according to the rules above:\n"
        f"UTTERANCE: {utterance}\n\n"
        "TASK:\n"
        "- For EACH feature in the list below, decide whether it is present (1) or absent (0) IN THIS UTTERANCE.\n"
        "- Treat each feature as a separate, small, task-grounded question about ONE semantic/grammatical element "
        "(e.g., possession, habitual aspect, negation configuration, WH word order, tense/aspect of the main verb).\n"
        "- Apply the procedure before making each binary decision.\n"
        f"{extra_cot_line}\n"
        f"FEATURE KEYS (in required order):\n[{feature_list_str}]\n\n"
        f"{output_format}"
    )


def build_messages(
    utterance: str,
    features: list[str],
    base_feature_block: str,
    instruction_type: str,
    include_examples_in_block: bool,
    use_context: bool = False,
    left_context: str | None = None,
    right_context: str | None = None,
    use_icl_examples: bool = False,
    dialect_legitimacy: bool = False,
    self_verification: bool = False,
    require_rationales: bool = True,
) -> list[dict]:
    """
    Build the full messages list for the OpenAI ChatCompletion API.
    - instruction_type: one of ["zero_shot", "icl", "zero_shot_cot", "few_shot_cot"]
    - include_examples_in_block: whether the feature block should retain + Example / - Miss lines.
      NOTE: For zero_shot / zero_shot_cot we forcibly strip examples regardless of this flag.
    - use_icl_examples: if True, append a separate FEW-SHOT section before the response instructions.
    """

    system_msg = build_system_msg(
        instruction_type=instruction_type,
        dialect_legitimacy=dialect_legitimacy,
        self_verification=self_verification,
    )

    # Enforce: zero-shot conditions never get examples embedded in the feature block
    effective_include_examples = include_examples_in_block
    if instruction_type in ["zero_shot", "zero_shot_cot"]:
        effective_include_examples = False

    if effective_include_examples:
        feature_block = base_feature_block
    else:
        feature_block = strip_examples_from_block(base_feature_block)

    parts = [feature_block]

    if use_icl_examples:
        parts.append(ICL_EXAMPLES_BLOCK)

    response_instructions = build_system_response_instructions(
        utterance=utterance,
        features=features,
        instruction_type=instruction_type,
        require_rationales=require_rationales,
    )
    parts.append(response_instructions)

    user_msg = {
        "role": "user",
        "content": "\n\n".join(parts),
    }
    return [system_msg, user_msg]


# -------------------- USAGE SUMMARY --------------------

def print_final_usage_summary():
    total_tokens = total_input_tokens + total_output_tokens

    print("\n===== FINAL USAGE SUMMARY =====")
    print(f"Total API Calls: {api_call_count}")
    print(f"Total Input Tokens: {total_input_tokens}")
    print(f"Total Output Tokens: {total_output_tokens}")
    print(f"Total Tokens: {total_tokens}")
    print("===================================\n")


def parse_output_json(raw_str: str, features: list[str]):
    """
    Supports BOTH:
    - Nested JSON: {feat: {"value": 0/1, "rationale": "..."}}
    - Flat JSON:   {feat: 0/1}
    """
    data = json.loads(raw_str)

    vals = {}
    rats = {}
    for feat in features:
        entry = data.get(feat, {})
        if isinstance(entry, (int, float)):
            vals[feat] = int(entry)
            rats[feat] = ""
        elif isinstance(entry, dict):
            vals[feat] = int(entry.get("value", 0))
            rats[feat] = str(entry.get("rationale", "")).strip()
        else:
            # Fallback: treat as missing / 0
            vals[feat] = 0
            rats[feat] = ""
    return vals, rats


# -------------------- GPT QUERY --------------------

def query_gpt(
    client: OpenAI,
    enc_obj,
    sentence: str,
    *,
    features: list[str],
    base_feature_block: str,
    instruction_type: str,
    include_examples_in_block: bool,
    use_context: bool = False,
    left_context: str | None = None,
    right_context: str | None = None,
    dialect_legitimacy: bool = False,
    self_verification: bool = False,
    require_rationales: bool = True,
    max_retries: int = 15,
    base_delay: int = 3,
) -> str | None:
    global api_call_count, total_input_tokens, total_output_tokens

    use_icl_examples = instruction_type in ["icl", "few_shot_cot"]

    messages = build_messages(
        utterance=sentence,
        features=features,
        base_feature_block=base_feature_block,
        instruction_type=instruction_type,
        include_examples_in_block=include_examples_in_block,
        use_context=use_context,
        left_context=left_context,
        right_context=right_context,
        use_icl_examples=use_icl_examples,
        dialect_legitimacy=dialect_legitimacy,
        self_verification=self_verification,
        require_rationales=require_rationales,
    )


    input_tokens = sum(len(enc_obj.encode(msg["content"])) for msg in messages)
    total_input_tokens += input_tokens

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL_NAME,
                messages=messages,
            )

            output_text = resp.choices[0].message.content
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


# -------------------- MAIN --------------------

def main():
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Run GPT Experiments for AAE feature annotation.")
    parser.add_argument("--file", type=str, help="Input Excel file path", required=True)
    parser.add_argument("--sheet", type=str, help="Sheet name to write GPT predictions into", required=True)
    parser.add_argument("--extended", action="store_true", help="Use extended feature set (NEW_FEATURE_BLOCK + EXTENDED_FEATURES)")
    parser.add_argument("--context", action="store_true", help="(Optional) Use prev/next sentence context from Gold sheet if available (currently not injected).")
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
        help="If set, keep + Example / - Miss lines inside the feature block (non-zero-shot conditions only).",
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
        help="If set, request flat JSON labels (0/1 only) with NO rationales.",
    )
    parser.add_argument("--output_dir", type=str, help="Output directory for CSV/Excel results", required=True)
    args = parser.parse_args()

    file_title = os.path.splitext(os.path.basename(args.file))[0]
    out_dir = os.path.join(args.output_dir, file_title)
    os.makedirs(out_dir, exist_ok=True)

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
    enc_obj = tiktoken.get_encoding("p50k_base")

    USE_EXTENDED = args.extended
    CURRENT_FEATURES = EXTENDED_FEATURES if USE_EXTENDED else MASIS_FEATURES
    BASE_FEATURE_BLOCK = NEW_FEATURE_BLOCK if USE_EXTENDED else MASIS_FEATURE_BLOCK

    include_examples_in_block = args.block_examples
    require_rationales = not args.labels_only

    preds_path = os.path.join(out_dir, args.sheet + "_predictions.csv")
    rats_path = os.path.join(out_dir, args.sheet + "_rationales.csv")

    preds_header = ["sentence"] + CURRENT_FEATURES
    rats_header = ["sentence"] + CURRENT_FEATURES

    if not os.path.exists(preds_path):
        with open(preds_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(preds_header)

    if not os.path.exists(rats_path):
        with open(rats_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(rats_header)

    results = []
    rationale_rows = []

    # Iterating through sentences to evaluate
    for idx, sentence in enumerate(tqdm(eval_sentences, desc="Evaluating sentences")):
        left = None
        right = None
        if args.context:
            if "prev_sentence" in gold_df.columns:
                left = gold_df.loc[idx, "prev_sentence"]
            if "next_sentence" in gold_df.columns:
                right = gold_df.loc[idx, "next_sentence"]

        raw = query_gpt(
            client,
            enc_obj,
            sentence,
            features=CURRENT_FEATURES,
            base_feature_block=BASE_FEATURE_BLOCK,
            instruction_type=args.instruction_type,
            include_examples_in_block=include_examples_in_block,
            use_context=args.context,
            left_context=left,
            right_context=right,
            dialect_legitimacy=args.dialect_legitimacy,
            self_verification=args.self_verification,
            require_rationales=require_rationales,
        )

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
            rat_row[feat] = rats.get(feat, "")
        rationale_rows.append(rat_row)

        # Append line-level CSV for robustness against interruptions
        with open(preds_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([sentence] + [vals.get(feat, "") for feat in CURRENT_FEATURES])

        with open(rats_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([sentence] + [rats.get(feat, "") for feat in CURRENT_FEATURES])

    # Aggregate predictions & rationales into DataFrames
    predictions_df = pd.DataFrame(results)
    rationales_df = pd.DataFrame(rationale_rows)

    # Write back to the Excel file (replace target sheet)
    with pd.ExcelWriter(args.file, mode='a', if_sheet_exists='replace') as writer:
        predictions_df.to_excel(writer, sheet_name=args.sheet, index=False)
        rats_sheet_name = f"{args.sheet}_rationales"
        rationales_df.to_excel(writer, sheet_name=rats_sheet_name, index=False)

    print_final_usage_summary()


if __name__ == "__main__":
    main()
