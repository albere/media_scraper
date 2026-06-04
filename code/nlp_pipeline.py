"""
nlp_pipeline.py
Immigration News Corpus — NLP Feature Extraction Pipeline

Extracts all 5 linguistic features from scored immigration articles:
  1. Sentiment        (VADER compound, pos/neg/neu ratios)
  2. Dehumanisation   (tiered lexicon density per 1000 words)
  3. Moral outrage    (lexicon density per 1000 words)
  4. Modal verbs      (certainty_ratio, epistemic_ratio)
  5. Pronouns         (inclusion_index, ingroup_ratio)

Input:  Scored CSV files with columns: date, body, outlet, immigration_score
        (plus any extras like 'section' — these are ignored gracefully)
Output: Per-article CSV with all features appended

Usage:
    python nlp_pipeline.py --input data/scored/guardian_scored.csv --output results/guardian_features.csv
    python nlp_pipeline.py --input-dir data/scored/ --output results/all_features.csv
    python nlp_pipeline.py --input-dir data/scored/ --output-dir results/per_outlet/

Core extraction logic adapted from CBA786 scripts (extract_sentiment.py,
extract_dehumanisation.py, extract_moral_outrage.py, extract_modals.py,
extract_pronouns.py). Demo/plotting/merge logic removed; I/O adapted
for news CSV format.
"""

import argparse
import csv
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

# Minimum immigration_score to include (set to 0 to skip filtering)
IMMIGRATION_THRESHOLD = 0.5

# Minimum word count to process an article
MIN_WORDS = 50

# Sentiment: minimum sentence length (words) to score
MIN_SENTENCE_LEN = 10

# Columns to read from input CSVs
# The pipeline requires 'body' and 'date'; 'outlet' and 'immigration_score'
# are used if present. All other columns are passed through unchanged.
REQUIRED_COLUMNS = {"body", "date"}

# Progress reporting interval
PROGRESS_EVERY = 500


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 1: SENTIMENT (VADER)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_sentiment(text, analyzer):
    """
    Sentence-level VADER scoring.

    Returns dict with:
        sentiment_compound  — mean compound score across sentences
        sentiment_pos_ratio — proportion of sentences with compound >= 0.05
        sentiment_neg_ratio — proportion of sentences with compound <= -0.05
        sentiment_neu_ratio — proportion in neutral band
        sentiment_n_sentences — number of sentences scored
    """
    raw_sentences = re.split(r'(?<=[.!?])\s+|\n+', text)

    compounds = []
    for sent in raw_sentences:
        sent = sent.strip()
        if len(sent.split()) < MIN_SENTENCE_LEN:
            continue
        compound = analyzer.polarity_scores(sent)['compound']
        compounds.append(compound)

    if not compounds:
        return {
            "sentiment_compound": None,
            "sentiment_pos_ratio": None,
            "sentiment_neg_ratio": None,
            "sentiment_neu_ratio": None,
            "sentiment_n_sentences": 0,
        }

    n = len(compounds)
    pos = sum(1 for c in compounds if c >= 0.05)
    neg = sum(1 for c in compounds if c <= -0.05)

    return {
        "sentiment_compound":    round(sum(compounds) / n, 4),
        "sentiment_pos_ratio":   round(pos / n, 4),
        "sentiment_neg_ratio":   round(neg / n, 4),
        "sentiment_neu_ratio":   round((n - pos - neg) / n, 4),
        "sentiment_n_sentences": n,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 2: DEHUMANISATION MARKERS
# ═══════════════════════════════════════════════════════════════════════════════

DEHUM_LEXICON = {
    # ── TIER 1: Core dehumanisation ──
    #
    # Lexicon v2: expanded from CBA786 parliamentary lexicon to cover
    # news register. Additions validated empirically against 5,000-article
    # Express sample. Terms marked [v2] are new additions.

    # Mass/water metaphors — reduce people to undifferentiated natural forces
    "flood":        {"tier": 1, "cat": "mass_metaphor"},
    "flooded":      {"tier": 1, "cat": "mass_metaphor"},
    "flooding":     {"tier": 1, "cat": "mass_metaphor"},
    "flooded in":   {"tier": 1, "cat": "mass_metaphor"},      # [v2]
    "floodgates":   {"tier": 1, "cat": "mass_metaphor"},      # [v2]
    "open the floodgates": {"tier": 1, "cat": "mass_metaphor"},  # [v2]
    "swarm":        {"tier": 1, "cat": "mass_metaphor"},
    "swarming":     {"tier": 1, "cat": "mass_metaphor"},
    "hordes":       {"tier": 1, "cat": "mass_metaphor"},
    "horde":        {"tier": 1, "cat": "mass_metaphor"},
    "wave":         {"tier": 1, "cat": "mass_metaphor"},
    "waves":        {"tier": 1, "cat": "mass_metaphor"},
    "invasion":     {"tier": 1, "cat": "mass_metaphor"},
    "invading":     {"tier": 1, "cat": "mass_metaphor"},
    "invaders":     {"tier": 1, "cat": "mass_metaphor"},
    "tide":         {"tier": 1, "cat": "mass_metaphor"},
    "tidal":        {"tier": 1, "cat": "mass_metaphor"},
    "influx":       {"tier": 1, "cat": "mass_metaphor"},
    "inundated":    {"tier": 1, "cat": "mass_metaphor"},
    "inundation":   {"tier": 1, "cat": "mass_metaphor"},
    "surge":        {"tier": 1, "cat": "mass_metaphor"},
    "swamped":      {"tier": 1, "cat": "mass_metaphor"},
    "overwhelm":    {"tier": 1, "cat": "mass_metaphor"},
    "overwhelming": {"tier": 1, "cat": "mass_metaphor"},
    "overwhelmed":  {"tier": 1, "cat": "mass_metaphor"},
    "overrun":      {"tier": 1, "cat": "mass_metaphor"},      # [v2]
    "overrunning":  {"tier": 1, "cat": "mass_metaphor"},      # [v2]
    "deluge":       {"tier": 1, "cat": "mass_metaphor"},      # [v2]
    "onslaught":    {"tier": 1, "cat": "mass_metaphor"},      # [v2]
    "besieged":     {"tier": 1, "cat": "mass_metaphor"},      # [v2]
    "pouring in":   {"tier": 1, "cat": "mass_metaphor"},      # [v2]
    "poured in":    {"tier": 1, "cat": "mass_metaphor"},      # [v2]
    "flocking":     {"tier": 1, "cat": "mass_metaphor"},      # [v2]
    "mass migration":    {"tier": 1, "cat": "mass_metaphor"}, # [v2]
    "mass immigration":  {"tier": 1, "cat": "mass_metaphor"}, # [v2]
    "open door":    {"tier": 1, "cat": "mass_metaphor"},      # [v2]
    "open borders": {"tier": 1, "cat": "mass_metaphor"},      # [v2]

    # Criminalisation / status denial
    "illegal immigrant":  {"tier": 1, "cat": "criminalisation"},
    "illegal immigrants": {"tier": 1, "cat": "criminalisation"},
    "illegal migrant":    {"tier": 1, "cat": "criminalisation"},
    "illegal migrants":   {"tier": 1, "cat": "criminalisation"},
    "illegal migration":  {"tier": 1, "cat": "criminalisation"},
    "illegal entry":      {"tier": 1, "cat": "criminalisation"},
    "illegal entrant":    {"tier": 1, "cat": "criminalisation"},
    "illegal entrants":   {"tier": 1, "cat": "criminalisation"},
    "illegal asylum":     {"tier": 1, "cat": "criminalisation"},
    "illegal alien":      {"tier": 1, "cat": "criminalisation"},
    "illegal aliens":     {"tier": 1, "cat": "criminalisation"},
    "illegals":           {"tier": 1, "cat": "criminalisation"},
    "bogus":              {"tier": 1, "cat": "criminalisation"},
    "bogus asylum":       {"tier": 1, "cat": "criminalisation"},
    "sneak":              {"tier": 1, "cat": "criminalisation"},  # [v2]
    "sneaked":            {"tier": 1, "cat": "criminalisation"},  # [v2]
    "sneaking":           {"tier": 1, "cat": "criminalisation"},  # [v2]
    "sham marriage":      {"tier": 1, "cat": "criminalisation"},  # [v2]
    "sham marriages":     {"tier": 1, "cat": "criminalisation"},  # [v2]
    "fake passport":      {"tier": 1, "cat": "criminalisation"},  # [v2]
    "fake passports":     {"tier": 1, "cat": "criminalisation"},  # [v2]
    "forged documents":   {"tier": 1, "cat": "criminalisation"},  # [v2]
    "clandestine":        {"tier": 1, "cat": "criminalisation"},  # [v2]

    # Criminalisation — smuggling/trafficking framing
    # These criminalise the process of immigration regardless of whether
    # migrants are framed as perpetrators or victims. In immigration-filtered
    # text, these terms overwhelmingly refer to migrant smuggling.   [v2]
    "smuggled":           {"tier": 1, "cat": "criminalisation"},  # [v2]
    "smuggling":          {"tier": 1, "cat": "criminalisation"},  # [v2]
    "smugglers":          {"tier": 1, "cat": "criminalisation"},  # [v2]
    "people smuggling":   {"tier": 1, "cat": "criminalisation"},  # [v2]
    "people smugglers":   {"tier": 1, "cat": "criminalisation"},  # [v2]
    "criminal gangs":     {"tier": 1, "cat": "criminalisation"},  # [v2]
    "gang":               {"tier": 1, "cat": "criminalisation"},  # [v2]
    "gangs":              {"tier": 1, "cat": "criminalisation"},  # [v2]
    "trafficked":         {"tier": 1, "cat": "criminalisation"},  # [v2]
    "trafficking":        {"tier": 1, "cat": "criminalisation"},  # [v2]
    "crackdown":          {"tier": 1, "cat": "criminalisation"},  # [v2]

    # Economic burden framing
    "burden":             {"tier": 1, "cat": "burden_framing"},
    "burdens":            {"tier": 1, "cat": "burden_framing"},
    "scrounger":          {"tier": 1, "cat": "burden_framing"},
    "scroungers":         {"tier": 1, "cat": "burden_framing"},
    "sponger":            {"tier": 1, "cat": "burden_framing"},
    "spongers":           {"tier": 1, "cat": "burden_framing"},
    "taxpayer":           {"tier": 1, "cat": "burden_framing"},   # [v2]
    "taxpayers":          {"tier": 1, "cat": "burden_framing"},   # [v2]
    "handout":            {"tier": 1, "cat": "burden_framing"},   # [v2]
    "handouts":           {"tier": 1, "cat": "burden_framing"},   # [v2]
    "health tourism":     {"tier": 1, "cat": "burden_framing"},   # [v2]
    "health tourist":     {"tier": 1, "cat": "burden_framing"},   # [v2]
    "health tourists":    {"tier": 1, "cat": "burden_framing"},   # [v2]
    "benefit tourist":    {"tier": 1, "cat": "burden_framing"},   # [v2]
    "benefit tourists":   {"tier": 1, "cat": "burden_framing"},   # [v2]

    # ── TIER 2: Contextual threat/crisis terms ──
    "crisis":        {"tier": 2, "cat": "threat_framing"},
    "threat":        {"tier": 2, "cat": "threat_framing"},
    "threats":       {"tier": 2, "cat": "threat_framing"},
    "strain":        {"tier": 2, "cat": "threat_framing"},
    "unsustainable": {"tier": 2, "cat": "threat_framing"},
    "uncontrolled":  {"tier": 2, "cat": "threat_framing"},
    "out of control": {"tier": 2, "cat": "threat_framing"},

    "dangerous individual":  {"tier": 2, "cat": "threat_framing"},
    "dangerous individuals": {"tier": 2, "cat": "threat_framing"},
    "dangerous person":      {"tier": 2, "cat": "threat_framing"},
    "dangerous people":      {"tier": 2, "cat": "threat_framing"},
    "dangerous criminal":    {"tier": 2, "cat": "threat_framing"},
    "dangerous criminals":   {"tier": 2, "cat": "threat_framing"},
    "dangerous foreign":     {"tier": 2, "cat": "threat_framing"},
}

# Pre-sort by length (longest first) so multi-word phrases match before substrings
_DEHUM_SORTED = sorted(DEHUM_LEXICON.keys(), key=len, reverse=True)

# Pre-compile regex patterns
_DEHUM_PATTERNS = {}
for _term in _DEHUM_SORTED:
    if " " in _term:
        _DEHUM_PATTERNS[_term] = re.compile(re.escape(_term))
    else:
        _DEHUM_PATTERNS[_term] = re.compile(r'\b' + re.escape(_term) + r'\b')


def extract_dehumanisation(text_lower, word_count):
    """
    Count dehumanisation lexicon hits.

    Parameters
    ----------
    text_lower : str — lowercased article body
    word_count : int — pre-computed word count

    Returns dict with:
        dehum_total, dehum_tier1, dehum_tier2,
        dehum_density, dehum_tier1_density, dehum_tier2_density,
        dehum_mass_metaphor_density, dehum_criminalisation_density,
        dehum_burden_density, dehum_threat_density
    """
    if word_count == 0:
        return {k: 0 for k in [
            "dehum_total", "dehum_tier1", "dehum_tier2",
            "dehum_density", "dehum_tier1_density", "dehum_tier2_density",
            "dehum_mass_metaphor_density", "dehum_criminalisation_density",
            "dehum_burden_density", "dehum_threat_density",
        ]}

    tier_counts = defaultdict(int)
    cat_counts = defaultdict(int)

    for term in _DEHUM_SORTED:
        meta = DEHUM_LEXICON[term]
        n_hits = len(_DEHUM_PATTERNS[term].findall(text_lower))
        if n_hits > 0:
            tier_counts[meta["tier"]] += n_hits
            cat_counts[meta["cat"]] += n_hits

    total = sum(tier_counts.values())
    k = word_count / 1000

    def d(n):
        return round(n / k, 4)

    return {
        "dehum_total":                  total,
        "dehum_tier1":                  tier_counts[1],
        "dehum_tier2":                  tier_counts[2],
        "dehum_density":                d(total),
        "dehum_tier1_density":          d(tier_counts[1]),
        "dehum_tier2_density":          d(tier_counts[2]),
        "dehum_mass_metaphor_density":  d(cat_counts["mass_metaphor"]),
        "dehum_criminalisation_density": d(cat_counts["criminalisation"]),
        "dehum_burden_density":         d(cat_counts["burden_framing"]),
        "dehum_threat_density":         d(cat_counts["threat_framing"]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 3: MORAL OUTRAGE LANGUAGE
# ═══════════════════════════════════════════════════════════════════════════════

OUTRAGE_LEXICON = {
    # Moral condemnation
    # Lexicon v2: expanded for news register. Terms marked [v2] are new.
    "shame": "moral_condemnation", "shameful": "moral_condemnation",
    "shameless": "moral_condemnation", "shaming": "moral_condemnation",
    "disgrace": "moral_condemnation", "disgraceful": "moral_condemnation",
    "disgraced": "moral_condemnation",
    "outrage": "moral_condemnation", "outrageous": "moral_condemnation",
    "outrageously": "moral_condemnation",
    "appalling": "moral_condemnation", "appalled": "moral_condemnation",
    "scandalous": "moral_condemnation", "scandal": "moral_condemnation",
    "unacceptable": "moral_condemnation",
    "reprehensible": "moral_condemnation", "repugnant": "moral_condemnation",
    "unconscionable": "moral_condemnation",
    "inhumane": "moral_condemnation", "inhuman": "moral_condemnation",
    "immoral": "moral_condemnation",
    "morally wrong": "moral_condemnation", "morally bankrupt": "moral_condemnation",
    "moral failure": "moral_condemnation", "moral outrage": "moral_condemnation",
    "criminal behaviour": "moral_condemnation", "criminal act": "moral_condemnation",
    "criminal acts": "moral_condemnation", "criminal conduct": "moral_condemnation",
    "criminal exploitation": "moral_condemnation", "criminal element": "moral_condemnation",
    "threat to our": "moral_condemnation", "threat to this": "moral_condemnation",
    "threat to national": "moral_condemnation", "existential threat": "moral_condemnation",
    "absurd": "moral_condemnation",            # [v2]
    "travesty": "moral_condemnation",          # [v2]

    # Institutional shame
    "betrayal": "institutional_shame", "betrayed": "institutional_shame",
    "betraying": "institutional_shame",
    "broken system": "institutional_shame",
    "broken promise": "institutional_shame", "broken promises": "institutional_shame",
    "deception": "institutional_shame", "deceptive": "institutional_shame",
    "deceit": "institutional_shame",
    "downright lies": "institutional_shame", "spreading lies": "institutional_shame",
    "telling lies": "institutional_shame", "these lies": "institutional_shame",
    "those lies": "institutional_shame", "lied": "institutional_shame",
    "fiasco": "institutional_shame",           # [v2]
    "farce": "institutional_shame",            # [v2]
    "shambles": "institutional_shame",         # [v2]

    # Outrage intensifiers
    "unprecedented": "outrage_intensifier",
    "deeply troubling": "outrage_intensifier", "deeply concerned": "outrage_intensifier",
    "deeply worrying": "outrage_intensifier", "deeply alarming": "outrage_intensifier",
    "deeply shameful": "outrage_intensifier", "gravely concerned": "outrage_intensifier",
    "utterly": "outrage_intensifier",
    "wholly unacceptable": "outrage_intensifier",
    "simply unacceptable": "outrage_intensifier",
    "morally unacceptable": "outrage_intensifier",
    "staggering": "outrage_intensifier",       # [v2]
    "shocking": "outrage_intensifier",         # [v2]
    "damning": "outrage_intensifier",          # [v2]

    # Tabloid outrage vocabulary — news-register terms for expressing
    # moral/political fury. Absent from parliamentary speech but
    # high-frequency in tabloid immigration coverage.              [v2]
    "fury": "tabloid_outrage",                 # [v2]
    "furious": "tabloid_outrage",              # [v2]
    "slammed": "tabloid_outrage",              # [v2]
    "blasted": "tabloid_outrage",              # [v2]
    "backlash": "tabloid_outrage",             # [v2]
    "uproar": "tabloid_outrage",               # [v2]
    "chaos": "tabloid_outrage",                # [v2]
    "chaotic": "tabloid_outrage",              # [v2]
    "catastrophe": "tabloid_outrage",          # [v2]
    "catastrophic": "tabloid_outrage",         # [v2]
    "disastrous": "tabloid_outrage",           # [v2]
    "madness": "tabloid_outrage",              # [v2]
    "lunacy": "tabloid_outrage",               # [v2]
}

_OUTRAGE_SORTED = sorted(OUTRAGE_LEXICON.keys(), key=len, reverse=True)

_OUTRAGE_PATTERNS = {}
for _term in _OUTRAGE_SORTED:
    if " " in _term:
        _OUTRAGE_PATTERNS[_term] = re.compile(re.escape(_term))
    else:
        _OUTRAGE_PATTERNS[_term] = re.compile(r'\b' + re.escape(_term) + r'\b')


def extract_moral_outrage(text_lower, word_count):
    """
    Count moral outrage lexicon hits.

    Returns dict with:
        outrage_total, outrage_density,
        outrage_condemnation_density, outrage_shame_density,
        outrage_intensifier_density, outrage_tabloid_density
    """
    if word_count == 0:
        return {k: 0 for k in [
            "outrage_total", "outrage_density",
            "outrage_condemnation_density", "outrage_shame_density",
            "outrage_intensifier_density", "outrage_tabloid_density",
        ]}

    cat_counts = defaultdict(int)

    for term in _OUTRAGE_SORTED:
        cat = OUTRAGE_LEXICON[term]
        n_hits = len(_OUTRAGE_PATTERNS[term].findall(text_lower))
        if n_hits > 0:
            cat_counts[cat] += n_hits

    total = sum(cat_counts.values())
    k = word_count / 1000

    def d(n):
        return round(n / k, 4)

    return {
        "outrage_total":                total,
        "outrage_density":              d(total),
        "outrage_condemnation_density": d(cat_counts["moral_condemnation"]),
        "outrage_shame_density":        d(cat_counts["institutional_shame"]),
        "outrage_intensifier_density":  d(cat_counts["outrage_intensifier"]),
        "outrage_tabloid_density":      d(cat_counts["tabloid_outrage"]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURES 4 & 5: MODALS + PRONOUNS (spaCy — single pass)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Modal marker sets ──

CERTAINTY_MARKERS = {
    "will", "must", "shall",
    "definitely", "certainly", "inevitably", "undoubtedly",
    "absolutely", "clearly", "obviously", "always", "necessarily",
}

HEDGING_MARKERS = {
    "might", "could",
    "perhaps", "possibly", "probably", "arguably", "seemingly",
    "potentially", "presumably", "apparently", "likely", "unlikely",
    "sometimes", "often",
}

AMBIGUOUS_MODAL_HEDGE = {"may", "would"}

EPISTEMIC_CERTAINTY = CERTAINTY_MARKERS - {"will"}
# 'would' is in AMBIGUOUS set, not HEDGING_MARKERS, so epistemic hedging
# is HEDGING_MARKERS + 'may' (handled in the loop)

# ── Pronoun sets ──

INGROUP  = {"we", "us", "our", "ours", "ourselves"}
OUTGROUP = {"they", "them", "their", "theirs", "themselves"}


def _is_modal_use(token):
    """
    Heuristic: return True if token is functioning as a modal verb,
    not as a proper noun (e.g. "May" the surname).
    """
    if token.is_sent_start:
        return True
    if token.text[0].isupper():
        prev = token.nbor(-1) if token.i > 0 else None
        if prev is not None:
            if prev.lower_ in {"mr", "mrs", "ms", "miss", "dr", "prof",
                                "hon", "right", "secretary", "minister",
                                "theresa", "prime"}:
                return False
    return True


def extract_spacy_features(doc):
    """
    Single-pass spaCy extraction for modals and pronouns.

    Returns dict with:
        Modal features:
            modal_certainty_count, modal_hedging_count, modal_word_count,
            modal_certainty_per1k, modal_hedging_per1k,
            modal_certainty_ratio, modal_epistemic_ratio,
            modal_balance, modal_epi_balance
        Pronoun features:
            pron_ingroup_count, pron_outgroup_count,
            pron_ingroup_per1k, pron_outgroup_per1k,
            pron_inclusion_index, pron_ingroup_ratio
    """
    certainty = hedging = epi_cert = epi_hedge = 0
    ingroup = outgroup = 0
    words = 0

    for token in doc:
        if not token.is_alpha:
            continue
        words += 1
        lower = token.lower_

        # ── Modals ──
        if lower in CERTAINTY_MARKERS:
            certainty += 1
            if lower in EPISTEMIC_CERTAINTY:
                epi_cert += 1
        elif lower in HEDGING_MARKERS:
            hedging += 1
            epi_hedge += 1
        elif lower in AMBIGUOUS_MODAL_HEDGE:
            if _is_modal_use(token):
                hedging += 1
                if lower == "may":
                    epi_hedge += 1

        # ── Pronouns ──
        if lower in INGROUP:
            ingroup += 1
        elif lower in OUTGROUP:
            outgroup += 1

    per1k = 1000 / words if words > 0 else 0

    # Modal metrics
    total_modal = certainty + hedging
    epi_total = epi_cert + epi_hedge
    certainty_ratio = certainty / total_modal if total_modal > 0 else float("nan")
    epistemic_ratio = epi_cert / epi_total if epi_total > 0 else float("nan")

    def _r(x):
        try:
            return round(x, 4) if x == x else x  # NaN check
        except TypeError:
            return x

    # Pronoun metrics
    pron_total = ingroup + outgroup
    ingroup_ratio = ingroup / pron_total if pron_total > 0 else float("nan")

    return {
        # Modals
        "modal_certainty_count":  certainty,
        "modal_hedging_count":    hedging,
        "modal_word_count":       words,
        "modal_certainty_per1k":  _r(certainty * per1k),
        "modal_hedging_per1k":    _r(hedging * per1k),
        "modal_certainty_ratio":  _r(certainty_ratio),
        "modal_epistemic_ratio":  _r(epistemic_ratio),
        "modal_balance":          _r((certainty - hedging) * per1k),
        "modal_epi_balance":      _r((epi_cert - epi_hedge) * per1k),
        # Pronouns
        "pron_ingroup_count":     ingroup,
        "pron_outgroup_count":    outgroup,
        "pron_ingroup_per1k":     _r(ingroup * per1k),
        "pron_outgroup_per1k":    _r(outgroup * per1k),
        "pron_inclusion_index":   _r((ingroup - outgroup) * per1k),
        "pron_ingroup_ratio":     _r(ingroup_ratio),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def process_article(text, analyzer, nlp):
    """
    Run all 5 feature extractors on a single article body.
    Returns a flat dict of all features.
    """
    text_lower = text.lower()
    word_count = len(text_lower.split())

    # Feature 1: Sentiment
    features = extract_sentiment(text, analyzer)

    # Feature 2: Dehumanisation
    features.update(extract_dehumanisation(text_lower, word_count))

    # Feature 3: Moral outrage
    features.update(extract_moral_outrage(text_lower, word_count))

    # Features 4+5: Modals + Pronouns (single spaCy pass)
    doc = nlp(text)
    features.update(extract_spacy_features(doc))

    # Add word count (from whitespace split, consistent with lexicon features)
    features["word_count"] = word_count

    return features


def run_pipeline(input_paths, output_path, threshold=IMMIGRATION_THRESHOLD):
    """
    Main pipeline: read scored CSVs, filter, extract features, save.
    """
    # ── Load models ──
    print("Loading models...")
    analyzer = SentimentIntensityAnalyzer()

    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        print("  spaCy: en_core_web_sm loaded")
    except OSError:
        print("  spaCy: en_core_web_sm not found, using blank English model")
        print("  (proper noun disambiguation for modals will be reduced)")
        nlp = spacy.blank("en")

    nlp.max_length = 3_000_000

    # ── Read input ──
    frames = []
    for p in input_paths:
        print(f"Reading {p} ... ", end="", flush=True)
        df = pd.read_csv(p, low_memory=False)

        # Check required columns
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            print(f"SKIP — missing columns: {missing}")
            continue

        # Infer outlet from filename if column not present
        if "outlet" not in df.columns:
            stem = Path(p).stem.lower()
            if "guardian" in stem:
                df["outlet"] = "guardian"
            elif "express" in stem:
                df["outlet"] = "express"
            elif "independent" in stem:
                df["outlet"] = "independent"
            elif "mirror" in stem:
                df["outlet"] = "mirror"
            else:
                df["outlet"] = stem
            print(f"(inferred outlet: {df['outlet'].iloc[0]}) ", end="")

        print(f"{len(df)} articles")
        frames.append(df)

    if not frames:
        print("ERROR: No valid input files found.")
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    print(f"\nTotal articles loaded: {len(df)}")

    # ── Filter by immigration score ──
    if threshold > 0 and "immigration_score" in df.columns:
        before = len(df)
        df = df[df["immigration_score"] >= threshold].copy()
        print(f"Filtered to immigration_score >= {threshold}: {len(df)} articles "
              f"(dropped {before - len(df)})")
    elif threshold > 0:
        print("WARNING: No 'immigration_score' column found — skipping threshold filter")

    # ── Filter by minimum word count ──
    df = df[df["body"].notna()].copy()
    df["_wc"] = df["body"].str.split().str.len()
    before = len(df)
    df = df[df["_wc"] >= MIN_WORDS].copy()
    if before - len(df) > 0:
        print(f"Dropped {before - len(df)} articles below {MIN_WORDS} words")
    df.drop(columns=["_wc"], inplace=True)

    print(f"Articles to process: {len(df)}")

    # ── Extract features ──
    print(f"\nExtracting features (reporting every {PROGRESS_EVERY} articles)...")
    t0 = time.time()
    results = []

    for i, (idx, row) in enumerate(df.iterrows()):
        features = process_article(row["body"], analyzer, nlp)
        results.append(features)

        if (i + 1) % PROGRESS_EVERY == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(df) - i - 1) / rate
            print(f"  [{i + 1:,}/{len(df):,}] {rate:.1f} articles/sec | "
                  f"~{remaining / 60:.1f} min remaining")

    elapsed = time.time() - t0
    print(f"\nExtraction complete: {len(results):,} articles in {elapsed:.1f}s "
          f"({len(results) / elapsed:.1f} articles/sec)")

    # ── Merge features with original data ──
    feature_df = pd.DataFrame(results)

    # Select columns to keep from original (drop body to save space)
    keep_cols = [c for c in df.columns if c != "body"]
    out = pd.concat([df[keep_cols].reset_index(drop=True), feature_df], axis=1)

    # ── Save ──
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"\n✓ Saved: {output_path} ({len(out)} rows, {len(out.columns)} columns)")

    # ── Summary ──
    print("\n── FEATURE SUMMARY ──")
    outlets = out["outlet"].unique() if "outlet" in out.columns else ["all"]
    for outlet in sorted(outlets):
        subset = out[out["outlet"] == outlet] if "outlet" in out.columns else out
        print(f"\n  {outlet.upper()} (n={len(subset)})")
        print(f"    Sentiment compound:   mean={subset['sentiment_compound'].mean():.4f}  "
              f"std={subset['sentiment_compound'].std():.4f}")
        print(f"    Dehum density:        mean={subset['dehum_density'].mean():.4f}  "
              f"std={subset['dehum_density'].std():.4f}")
        print(f"    Outrage density:      mean={subset['outrage_density'].mean():.4f}  "
              f"std={subset['outrage_density'].std():.4f}")
        print(f"    Certainty ratio:      mean={subset['modal_certainty_ratio'].mean():.4f}  "
              f"std={subset['modal_certainty_ratio'].std():.4f}")
        print(f"    Epistemic ratio:      mean={subset['modal_epistemic_ratio'].mean():.4f}  "
              f"std={subset['modal_epistemic_ratio'].std():.4f}")
        print(f"    Inclusion index:      mean={subset['pron_inclusion_index'].mean():.4f}  "
              f"std={subset['pron_inclusion_index'].std():.4f}")

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NLP Feature Extraction Pipeline — Immigration News Corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", nargs="+",
        help="One or more scored CSV files to process",
    )
    input_group.add_argument(
        "--input-dir",
        help="Directory containing scored CSV files (processes all *.csv)",
    )

    parser.add_argument(
        "--output", required=True,
        help="Output CSV path (single file with all results)",
    )
    parser.add_argument(
        "--threshold", type=float, default=IMMIGRATION_THRESHOLD,
        help=f"Minimum immigration_score to include (default: {IMMIGRATION_THRESHOLD}; "
             f"set to 0 to disable filtering)",
    )
    parser.add_argument(
        "--no-filter", action="store_true",
        help="Skip immigration score filtering entirely (equivalent to --threshold 0)",
    )

    args = parser.parse_args()

    # Resolve input files
    if args.input:
        input_paths = [Path(p) for p in args.input]
    else:
        input_dir = Path(args.input_dir)
        input_paths = sorted(input_dir.glob("*.csv"))
        if not input_paths:
            print(f"ERROR: No CSV files found in {input_dir}")
            sys.exit(1)

    # Check files exist
    for p in input_paths:
        if not p.exists():
            print(f"ERROR: File not found: {p}")
            sys.exit(1)

    threshold = 0 if args.no_filter else args.threshold

    print("=" * 65)
    print("NLP Feature Extraction Pipeline")
    print(f"  Input:     {len(input_paths)} file(s)")
    print(f"  Output:    {args.output}")
    print(f"  Threshold: {threshold}")
    print("=" * 65 + "\n")

    run_pipeline(input_paths, args.output, threshold=threshold)


if __name__ == "__main__":
    main()
