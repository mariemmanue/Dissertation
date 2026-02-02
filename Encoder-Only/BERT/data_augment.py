#!/usr/bin/env python
import csv
import random
import re

random.seed(42)

# --------- CONFIG ---------
INPUT_TSV  = "train.tsv"
OUTPUT_TSV = "train_augmented.tsv"

# Column name of the sentence
TEXT_COL = "example"

# 17 feature column names in order (must match your header)
FEATURE_COLS = [
    "zero-poss", "zero-copula", "double-tense", "be-construction",
    "resultant-done", "finna", "come", "double-modal",
    "multiple-neg", "neg-inversion", "n-inv-neg-concord", "aint",
    "zero-3sg-pres-s", "is-was-gen", "zero-pl-s", "double-object", "wh-qu",
]

# How many augmented variants per original sentence
N_AUG_PER_SENT = 2

# Generic narrative prefixes / suffixes for *all* features
GENERIC_PREFIXES = [
    "You know, when I think back on it,",
    "Sometimes, late at night,",
    "Back when we stayed in Princeville,",
    "I remember this one time,",
    "Growing up, we used to say stuff like,",
    "When we was talking the other day,",
    "In conversations with my family,",
]

GENERIC_SUFFIXES = [
    "and that’s just how we talked back then.",
    "and nobody really thought nothing of it.",
    "and everybody knew exactly what I meant.",
    "and that’s how the story went in my family.",
    "and I still catch myself saying it that way now.",
    "and that’s the kind of thing we’d say on the porch.",
]

# Fillers that can go before or after the core span
FILLERS = [
    "you know", "I mean", "like", "for real", "honestly",
    "you feel me", "lowkey", "to be honest",
]

# Very *mild* noun/adj lists – used only OUTSIDE the core span
NOUNS = ["house", "church", "school", "store", "parade", "funeral", "job", "yard", "porch"]
ADJS  = ["little", "old", "big", "small", "nice", "pretty", "tired", "happy"]


def insert_random_filler(text: str) -> str:
    """Insert a filler near a comma or before the core clause boundary."""
    if random.random() < 0.5:
        return text  # sometimes skip

    filler = random.choice(FILLERS)

    # Try to insert after first comma
    if "," in text:
        parts = text.split(",", 1)
        return parts[0] + ", " + filler + "," + parts[1]

    # Else insert before last clause boundary marker
    for mark in [" and ", " but ", " so "]:
        idx = text.rfind(mark)
        if idx != -1:
            return text[:idx] + " " + filler + text[idx:]

    # Fallback: append at end
    return text + ", " + filler


def light_noun_adj_substitution(outside_core: str) -> str:
    """
    Very conservative: replace a single noun or adjective token with
    something from our lists, to add variety without touching AAE span.
    """
    tokens = outside_core.split()
    if not tokens:
        return outside_core

    # choose a random index
    i = random.randrange(len(tokens))
    tok = tokens[i].lower().strip(",.?!")

    # Heuristic: if looks plural, don't touch
    if tok.endswith("s") and tok not in ("is", "was"):
        return outside_core

    # Randomly decide whether to use noun or adj bank
    if random.random() < 0.5:
        replacement = random.choice(NOUNS)
    else:
        replacement = random.choice(ADJS)

    tokens[i] = replacement
    return " ".join(tokens)


def safe_augment_sentence(core: str) -> str:
    """
    Create one advanced, test-like sentence from a core clause, while
    trying hard not to change the core string itself.
    """
    prefix = random.choice(GENERIC_PREFIXES)
    suffix = random.choice(GENERIC_SUFFIXES)

    # Build base with prefix + core + suffix
    base = f"{prefix} {core} {suffix}"

    # Optionally insert a filler
    base = insert_random_filler(base)

    # Optionally lightly substitute outside the core span:
    # we will try to protect the exact core substring.
    if random.random() < 0.5:
        # Find core span and only modify outside it
        try:
            start = base.index(core)
            end = start + len(core)
            before = base[:start]
            middle = base[start:end]
            after = base[end:]

            before = light_noun_adj_substitution(before)
            after = light_noun_adj_substitution(after)

            base = before + middle + after
        except ValueError:
            # core not found exactly, just do nothing extra
            pass

    return base


def main():
    with open(INPUT_TSV, encoding="utf-8") as fin, \
         open(OUTPUT_TSV, "w", encoding="utf-8", newline="") as fout:

        reader = csv.DictReader(fin, delimiter="\t")
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("No header found in input TSV.")

        if TEXT_COL not in fieldnames:
            raise ValueError(f"Expected a '{TEXT_COL}' column in header, got: {fieldnames}")

        for f in FEATURE_COLS:
            if f not in fieldnames:
                raise ValueError(f"Feature column '{f}' missing from header.")

        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for row in reader:
            core = row[TEXT_COL].strip()
            if not core:
                continue

            # 1) write original
            writer.writerow(row)

            # 2) write augmented variants with same labels
            for _ in range(N_AUG_PER_SENT):
                aug = safe_augment_sentence(core)
                new_row = dict(row)
                new_row[TEXT_COL] = aug
                writer.writerow(new_row)

    print(f"Wrote augmented data to {OUTPUT_TSV}")


if __name__ == "__main__":
    main()
