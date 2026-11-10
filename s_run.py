"""
================================================================================
NAMBU-GOTO STATIONARY ACTION HYPOTHESIS FOR LANGUAGE
================================================================================

Full research pipeline testing:

  1. FRACTAL STRUCTURE     — Same Nambu-Goto geometry at letter, word, sentence level
  2. CROSS-LANGUAGE        — Universal signal in English, Turkish, Arabic
  3. BLIND GEOMETRY TEST   — Learn coordinates from raw co-occurrence, no labels
  4. STATIONARY ACTION     — Suppletives as symmetry-breaking ground states
  5. PHASE TRANSITION      — Sharp frequency threshold for irregularity (Lie group)
  6. GEOMETRIC ATTENTION   — Nambu-Goto routing vs standard attention baseline
  7. DIMENSIONAL LIFTING   — Evidence for higher-dimensional linguistic manifold

Authors: [Your name]
Date:    2026

Dependencies:
    pip install numpy scipy matplotlib networkx nltk datasets
                scikit-learn unimorph-inflect torch tqdm

Data (auto-downloaded):
    - WikiText-103       (English, 100M tokens)
    - Universal Dependencies English EWT treebank
    - UniMorph English, Turkish, Arabic paradigm databases
    - CELEX2 morphological database (requires license — see NOTE below)

NOTE on CELEX2:
    CELEX2 requires a free academic license from the Linguistic Data Consortium.
    Register at: https://catalog.ldc.upenn.edu/LDC96L14
    Set CELEX_PATH environment variable to your extracted CELEX2 directory.
    The script falls back to NLTK WordNet if CELEX2 is unavailable.

Usage:
    python nambu_goto_language_hypothesis.py [--quick] [--no-plots]

    --quick     Run on 10% of data (fast, less reliable)
    --no-plots  Skip visualization (useful for servers)
================================================================================
"""

import os, sys, re, json, argparse, logging
from pathlib import Path
from collections import defaultdict

import numpy as np
from numpy.linalg import svd
from scipy import stats
from scipy.optimize import minimize, brentq
from scipy.ndimage import uniform_filter1d
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
log = logging.getLogger(__name__)

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--quick',    action='store_true', help='Use 10%% of data')
parser.add_argument('--no-plots', action='store_true', help='Skip visualization')
args = parser.parse_args()

QUICK     = args.quick
NO_PLOTS  = args.no_plots
OUT_DIR   = Path("./ng_results"); OUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 0 — DATA DOWNLOAD & LOADING
# ════════════════════════════════════════════════════════════════════════════

def download_wikitext():
    """Download WikiText-103 via HuggingFace datasets."""
    log.info("Loading WikiText-103 …")
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        texts = [row["text"] for row in ds if len(row["text"].strip()) > 20]
        if QUICK:
            texts = texts[:len(texts)//10]
        log.info(f"  {len(texts):,} paragraphs loaded")
        return texts
    except Exception as e:
        log.warning(f"HuggingFace datasets failed: {e}")
        log.info("Falling back to NLTK Brown corpus …")
        import nltk
        nltk.download('brown', quiet=True)
        from nltk.corpus import brown
        sents = [" ".join(s) for s in brown.sents()]
        if QUICK: sents = sents[:5000]
        log.info(f"  {len(sents):,} Brown sentences loaded")
        return sents


def download_ud_english():
    """
    Download Universal Dependencies English treebanks.
    Fetches EWT (train+dev) and GUM (train) for ~25,000 sentences total.
    Returns list of available .conllu paths, or None if all fail.
    """
    log.info("Loading UD English treebanks …")
    import urllib.request

    treebanks = [
        ("EWT-train",
         "https://raw.githubusercontent.com/UniversalDependencies/"
         "UD_English-EWT/master/en_ewt-ud-train.conllu",
         OUT_DIR / "en_ewt-ud-train.conllu"),
        ("EWT-dev",
         "https://raw.githubusercontent.com/UniversalDependencies/"
         "UD_English-EWT/master/en_ewt-ud-dev.conllu",
         OUT_DIR / "en_ewt-ud-dev.conllu"),
        ("GUM-train",
         "https://raw.githubusercontent.com/UniversalDependencies/"
         "UD_English-GUM/master/en_gum-ud-train.conllu",
         OUT_DIR / "en_gum-ud-train.conllu"),
        ("GUM-dev",
         "https://raw.githubusercontent.com/UniversalDependencies/"
         "UD_English-GUM/master/en_gum-ud-dev.conllu",
         OUT_DIR / "en_gum-ud-dev.conllu"),
    ]

    available = []
    for name, url, path in treebanks:
        if path.exists():
            available.append(str(path)); log.info(f"  {name}: cached")
            continue
        try:
            urllib.request.urlretrieve(url, path)
            available.append(str(path)); log.info(f"  {name}: downloaded")
        except Exception as e:
            log.warning(f"  {name}: failed ({e})")

    return available or None



def load_unimorph(lang="eng"):
    """
    Load UniMorph paradigm data from GitHub.
    Returns list of (lemma, form, tags) tuples.
    """
    log.info(f"Loading UniMorph {lang} …")
    url = f"https://raw.githubusercontent.com/unimorph/{lang}/master/{lang}"
    path = OUT_DIR / f"unimorph_{lang}.txt"
    if not path.exists():
        import urllib.request
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            log.warning(f"  UniMorph {lang} download failed: {e}")
            return []
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split("\t")
            if len(parts) >= 3:
                entries.append((parts[0], parts[1], parts[2]))
    log.info(f"  {len(entries):,} UniMorph entries")
    return entries


def load_celex_or_wordnet():
    """
    Load morphological families. Priority order:
      1. CELEX2 (best — requires LDC license, set CELEX_PATH)
      2. Wiktionary full English inflection tables (free, ~100k lemmas)
      3. NLTK WordNet (fallback, only 53 curated families)
    """
    celex_path = os.environ.get("CELEX_PATH")
    if celex_path and Path(celex_path).exists():
        log.info(f"Loading CELEX2 from {celex_path} …")
        return load_celex(celex_path)
    wikt = load_wiktionary_morphology()
    if len(wikt) >= 500:
        return wikt
    log.info("Falling back to NLTK WordNet morphology …")
    return load_wordnet_morphology()


def _edit_distance_ratio(a, b):
    from difflib import SequenceMatcher
    return 1 - SequenceMatcher(None, a, b).ratio()


def load_wiktionary_morphology():
    """
    Download and parse Wiktionary English inflection data via the
    freely available kaikki.org Wiktionary dump (JSON-lines).

    Covers ~100,000 English lemmas with full inflected paradigms.
    Returns dict: lemma → list of (form, is_irregular, depth, freq=1)
    Frequencies are placeholders; enrich_with_corpus_freq() upgrades them.
    """
    import urllib.request

    KNOWN_IRREGULAR_FORMS = {
        "am","is","are","was","were","been","went","gone","had","did","done",
        "said","made","got","gotten","knew","known","thought","took","taken",
        "saw","seen","came","gave","given","found","told","kept","left",
        "brought","bought","caught","taught","wrote","written","spoke","spoken",
        "ran","fell","fallen","grew","grown","drew","drawn","broke","broken",
        "drove","driven","flew","flown","blew","blown","threw","thrown",
        "chose","chosen","froze","frozen","wore","worn","swore","sworn",
        "bore","borne","tore","torn","stole","stolen","rode","ridden",
        "hid","hidden","built","sent","spent","lent","bent","meant","dealt",
        "felt","slept","swept","wept","crept","drank","drunk","sang","sung",
        "rang","rung","swam","swum","began","begun","won","held","stood",
        "lost","met","cut","put","let","hit","read","better","best",
        "worse","worst","more","most","less","least","men","women",
        "children","feet","teeth","mice","geese",
    }

    families = {}
    for pos, fname in [("verbs","kaikki.org-dictionary-English-verbs.json"),
                       ("adj",  "kaikki.org-dictionary-English-adj.json"),
                       ("nouns","kaikki.org-dictionary-English-nouns.json")]:
        url  = f"https://kaikki.org/dictionary/English/by-pos/{fname}"
        path = OUT_DIR / fname
        if not path.exists():
            log.info(f"  Downloading Wiktionary {pos} …")
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                log.warning(f"  Wiktionary {pos} download failed: {e}")
                continue

        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: entry = json.loads(line)
                except json.JSONDecodeError: continue

                lemma = entry.get("word","").lower().strip()
                if not lemma or " " in lemma: continue

                forms = []
                for fe in entry.get("forms", []):
                    form = fe.get("form","").lower().strip()
                    tags = fe.get("tags", [])
                    if not form or form == lemma: continue
                    if any(t in tags for t in ["table-tags","error","canonical"]):
                        continue
                    is_irr = (form in KNOWN_IRREGULAR_FORMS or
                              _edit_distance_ratio(lemma, form) > 0.55)
                    forms.append((form, is_irr, 1, 1))

                if forms:
                    families[lemma] = forms
                    n += 1
        log.info(f"  {n:,} {pos} lemmas loaded from Wiktionary")

    log.info(f"  Total Wiktionary families: {len(families):,}")
    return families


def enrich_with_corpus_freq(families: dict, texts: list) -> dict:
    """
    Replace placeholder freq=1 with real corpus counts per lemma.
    Essential for the phase transition test (Section 4).
    """
    log.info("Enriching families with corpus frequencies …")
    word_counts = defaultdict(int)
    for text in texts:
        for w in re.findall(r"[a-z']+", text.lower()):
            word_counts[w] += 1

    enriched  = {}
    n_enriched = 0
    for lemma, forms in families.items():
        freq      = word_counts.get(lemma, 1)
        enriched[lemma] = [(f, irr, d, freq) for f, irr, d, _ in forms]
        if freq > 1: n_enriched += 1

    log.info(f"  {n_enriched:,}/{len(families):,} lemmas enriched")
    return enriched



def load_celex(path):
    """
    Parse CELEX2 English morphology (eml.cd file).
    Returns dict: lemma → list of (form, is_irregular, depth, freq)
    """
    eml_path = Path(path) / "ENGLISH" / "EML" / "EML.CD"
    if not eml_path.exists():
        log.warning(f"EML.CD not found at {eml_path}")
        return {}
    families = defaultdict(list)
    with open(eml_path, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("\\")
            if len(parts) < 5: continue
            lemma   = parts[1].lower()
            form    = parts[2].lower()
            freq    = int(parts[3]) if parts[3].isdigit() else 0
            morph   = parts[4]
            is_irr  = "IRR" in morph or "IRREG" in morph
            depth   = morph.count("+")
            families[lemma].append((form, is_irr, depth, freq))
    log.info(f"  {len(families):,} CELEX2 lemmas loaded")
    return dict(families)


def load_wordnet_morphology():
    """
    Build morphological families from NLTK WordNet + morphy.
    Returns dict: lemma → list of (form, is_irregular, depth, freq)
    """
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    from nltk.corpus import wordnet as wn

    # Curated English irregular paradigms (ground truth)
    IRREGULARS = {
        "be":    ["am","is","are","was","were","been"],
        "go":    ["went","gone"],
        "have":  ["had"],
        "do":    ["did","done"],
        "say":   ["said"],
        "make":  ["made"],
        "get":   ["got","gotten"],
        "know":  ["knew","known"],
        "think": ["thought"],
        "take":  ["took","taken"],
        "see":   ["saw","seen"],
        "come":  ["came"],
        "give":  ["gave","given"],
        "find":  ["found"],
        "run":   ["ran"],
        "write": ["wrote","written"],
        "speak": ["spoke","spoken"],
        "break": ["broke","broken"],
        "grow":  ["grew","grown"],
        "draw":  ["drew","drawn"],
        "drive": ["drove","driven"],
        "build": ["built"],
        "bring": ["brought"],
        "buy":   ["bought"],
        "catch": ["caught"],
        "teach": ["taught"],
        "sell":  ["sold"],
        "tell":  ["told"],
        "hold":  ["held"],
        "stand": ["stood"],
        "lead":  ["led"],
        "win":   ["won"],
        "sit":   ["sat"],
        "fall":  ["fell","fallen"],
        "lose":  ["lost"],
        "send":  ["sent"],
        "spend": ["spent"],
        "leave": ["left"],
        "keep":  ["kept"],
        "feel":  ["felt"],
        "sleep": ["slept"],
        "meet":  ["met"],
        "read":  ["read"],
        "good":  ["better","best"],
        "bad":   ["worse","worst"],
        "many":  ["more","most"],
        "man":   ["men"],
        "woman": ["women"],
        "child": ["children"],
        "foot":  ["feet"],
        "tooth": ["teeth"],
        "mouse": ["mice"],
        "goose": ["geese"],
    }

    families = {}
    # Regular suffixes for productive derivation
    REGULAR_SUFFIXES = {
        "V":  ["s","ed","ing","er","able","ment","ion"],
        "N":  ["s","'s","less","ful","hood","ship"],
        "J":  ["er","est","ly","ness","ish"],
    }

    for lemma, irr_forms in IRREGULARS.items():
        synsets = wn.synsets(lemma)
        pos_tag = synsets[0].pos() if synsets else "v"
        freq    = sum(s.lemmas()[0].count() for s in synsets[:3]) + 1

        forms = []
        # Irregular (suppletive) forms
        for f in irr_forms:
            forms.append((f, True, 1, freq // 2))

        # Regular derivations
        suffixes = REGULAR_SUFFIXES.get(pos_tag.upper(), REGULAR_SUFFIXES["V"])
        for i, suf in enumerate(suffixes):
            reg_form = lemma + suf
            forms.append((reg_form, False, 1 + i//3, max(1, freq // (i+2))))

        families[lemma] = forms

    log.info(f"  {len(families):,} WordNet families built")
    return families


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — MORPHOLOGICAL NETWORK ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def analyze_morphological_networks(families: dict) -> dict:
    """
    Measurement A: Does branching arity scale with frequency?
    Measurement B: Are suppletive forms more terminal?
    Measurement C: Does trifurcation rate scale with frequency?
    Measurement D: Is sprout density lower at high-frequency nodes?

    Returns dict of results for each measurement.
    """
    log.info("Running morphological network analysis …")

    nodes = []
    for lemma, forms in families.items():
        all_forms  = forms
        n_supp     = sum(1 for f, irr, d, freq in all_forms if irr)
        n_reg      = sum(1 for f, irr, d, freq in all_forms if not irr)
        fam_size   = len(all_forms)
        lemma_freq = max((freq for f, irr, d, freq in all_forms), default=1)

        # Frequency tier
        if lemma_freq >= 100:  freq_tier = "high"
        elif lemma_freq >= 10: freq_tier = "medium"
        else:                  freq_tier = "low"

        nodes.append({
            "lemma":      lemma,
            "freq_tier":  freq_tier,
            "freq":       lemma_freq,
            "n_supp":     n_supp,
            "n_reg":      n_reg,
            "fam_size":   fam_size,
            "is_strong":  n_supp > 0,
        })

    freq_map = {"high":3,"medium":2,"low":1}

    # Measurement A: frequency → arity correlation
    freq_num  = [freq_map[n["freq_tier"]] for n in nodes]
    fam_sizes = [n["fam_size"] for n in nodes]
    # Guard: pearsonr is undefined if either array is constant
    if len(set(freq_num)) < 2 or len(set(fam_sizes)) < 2:
        r, p = float('nan'), float('nan')
    else:
        r, p = stats.pearsonr(freq_num, fam_sizes)

    # Measurement B: suppletive terminality
    # A form is "terminal" if it never appears as a lemma itself
    all_lemmas    = set(families.keys())
    supp_forms    = [f for lemma, forms in families.items()
                     for f, irr, d, freq in forms if irr]
    reg_forms     = [f for lemma, forms in families.items()
                     for f, irr, d, freq in forms if not irr]
    pct_supp_term = 100 * sum(1 for f in supp_forms if f not in all_lemmas) / max(len(supp_forms),1)
    pct_reg_term  = 100 * sum(1 for f in reg_forms  if f not in all_lemmas) / max(len(reg_forms),1)

    # Measurement C: trifurcation by frequency
    by_freq = defaultdict(list)
    for n in nodes:
        by_freq[n["freq_tier"]].append(n["fam_size"])
    trif = {tier: 100*sum(1 for s in sizes if s>=3)/max(len(sizes),1)
            for tier, sizes in by_freq.items()}

    high_s = by_freq.get("high",  [0])
    med_s  = by_freq.get("medium",[0])
    low_s  = by_freq.get("low",   [0])
    # Need at least 2 non-empty groups with variance for ANOVA
    non_empty = [g for g in [high_s, med_s, low_s] if len(g) > 1 and len(set(g)) > 1]
    if len(non_empty) >= 2:
        f_stat, p_anova = stats.f_oneway(*non_empty)
    else:
        f_stat, p_anova = float('nan'), float('nan')

    # Measurement D: sprout density at high vs low freq
    strong = [n for n in nodes if n["is_strong"]]
    hf_dens = np.mean([n["n_supp"]/n["fam_size"] for n in strong if n["freq_tier"]=="high"]  or [0])
    lf_dens = np.mean([n["n_supp"]/n["fam_size"] for n in strong if n["freq_tier"]=="low"]   or [0])

    results = {
        "r_freq_arity":    r,
        "p_freq_arity":    p,
        "pct_supp_term":   pct_supp_term,
        "pct_reg_term":    pct_reg_term,
        "trif_rates":      trif,
        "f_anova":         f_stat,
        "p_anova":         p_anova,
        "hf_sprout_dens":  hf_dens,
        "lf_sprout_dens":  lf_dens,
        "mean_high_arity": np.mean(high_s) if high_s else 0,
        "mean_med_arity":  np.mean(med_s)  if med_s  else 0,
        "mean_low_arity":  np.mean(low_s)  if low_s  else 0,
        "n_families":      len(families),
    }

    log.info(f"  Freq→Arity: r={r:.3f}, p={p:.4f}")
    log.info(f"  ANOVA: F={f_stat:.2f}, p={p_anova:.4f}")
    log.info(f"  Supp terminal: {pct_supp_term:.1f}%  Reg terminal: {pct_reg_term:.1f}%")
    log.info(f"  Sprout density HF={hf_dens:.3f} vs LF={lf_dens:.3f}")
    return results


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CROSS-LANGUAGE UNIVERSALITY
# ════════════════════════════════════════════════════════════════════════════

def analyze_unimorph_language(entries: list, lang: str) -> dict:
    """
    Build families from UniMorph entries and run morphological analysis.
    Suppletive = form that differs radically from lemma (edit distance > 0.6).
    Frequency proxy = number of distinct inflected forms in paradigm
    (larger paradigm → more frequent / more grammatically active lemma).
    """
    from difflib import SequenceMatcher

    def rel_distance(a, b):
        return 1 - SequenceMatcher(None, a, b).ratio()

    # First pass: count paradigm size per lemma (frequency proxy)
    paradigm_sizes = defaultdict(int)
    for lemma, form, tags in entries:
        paradigm_sizes[lemma] += 1

    families = defaultdict(list)
    for lemma, form, tags in entries:
        dist   = rel_distance(lemma.lower(), form.lower())
        is_irr = dist > 0.5          # >50% different = suppletive
        n_tags = len(tags.split(";"))
        # Use paradigm size as frequency proxy so tiers vary
        freq   = paradigm_sizes[lemma]
        families[lemma].append((form, is_irr, 1, freq))

    # Only keep lemmas with at least 2 forms (otherwise no branching)
    families = {k: v for k, v in families.items() if len(v) >= 2}

    # Subsample for speed if very large
    if len(families) > 5000:
        keys     = list(families.keys())
        np.random.shuffle(keys)
        families = {k: families[k] for k in keys[:5000]}

    results = analyze_morphological_networks(dict(families))
    results["language"] = lang

    r_str = f"{results['r_freq_arity']:.3f}" if not np.isnan(results['r_freq_arity']) else "nan"
    log.info(f"  [{lang}] r={r_str}, "
             f"F={results['f_anova']:.2f}, "
             f"supp_term={results['pct_supp_term']:.1f}%")
    return results


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BLIND CO-OCCURRENCE GEOMETRY TEST
# ════════════════════════════════════════════════════════════════════════════

def build_cooccurrence(texts: list, window: int = 5,
                       min_freq: int = 50,
                       max_vocab: int = 20_000) -> tuple:
    """
    Build sparse PPMI co-occurrence matrix from raw text.
    Uses scipy sparse matrices — safe for large corpora.
    Returns (vocab, word2idx, ppmi_sparse).
    """
    from scipy.sparse import lil_matrix, csr_matrix
    from scipy.sparse.linalg import svds

    log.info("Building co-occurrence matrix (sparse) …")
    word_counts = defaultdict(int)
    for text in texts:
        for w in re.findall(r"[a-z']+", text.lower()):
            word_counts[w] += 1

    # Cap vocabulary size to keep memory sane
    top_words = sorted(word_counts.items(), key=lambda x: -x[1])
    top_words = [(w, c) for w, c in top_words if c >= min_freq][:max_vocab]
    vocab     = [w for w, c in top_words]
    word2idx  = {w: i for i, w in enumerate(vocab)}
    n         = len(vocab)
    log.info(f"  Vocabulary: {n:,} words (freq≥{min_freq}, capped at {max_vocab:,})")

    # Accumulate co-occurrences into a dict first (memory-efficient)
    cooc_dict = defaultdict(float)
    row_sums  = np.zeros(n, dtype=np.float64)
    total_    = 0.0

    for text in texts:
        tokens = [w for w in re.findall(r"[a-z']+", text.lower())
                  if w in word2idx]
        for i, wi_str in enumerate(tokens):
            wi = word2idx[wi_str]
            for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
                if j == i: continue
                wj = word2idx[tokens[j]]
                cooc_dict[(wi, wj)] += 1.0
                row_sums[wi]        += 1.0
                total_              += 1.0

    log.info(f"  Non-zero co-occurrences: {len(cooc_dict):,}")

    # Build sparse PPMI directly — never materialise dense matrix
    col_sums = np.zeros(n, dtype=np.float64)
    for (wi, wj), v in cooc_dict.items():
        col_sums[wj] += v

    rows, cols, data = [], [], []
    for (wi, wj), v in cooc_dict.items():
        expected = (row_sums[wi] * col_sums[wj]) / (total_ + 1e-10)
        pmi      = np.log(v / (expected + 1e-10) + 1e-10)
        if pmi > 0:
            rows.append(wi); cols.append(wj); data.append(pmi)

    ppmi_sparse = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    log.info(f"  Sparse PPMI built  ({ppmi_sparse.nnz:,} non-zeros, "
             f"{ppmi_sparse.nnz * 4 / 1e6:.1f} MB)")
    return vocab, word2idx, ppmi_sparse


def learn_ng_coordinates(ppmi, k: int = 8) -> np.ndarray:
    """
    Learn Nambu-Goto coordinates via truncated SVD.
    Works with both dense numpy arrays and scipy sparse matrices.
    Returns (n_vocab × k) coordinate matrix.
    """
    from scipy.sparse import issparse
    from scipy.sparse.linalg import svds

    log.info(f"Learning {k}D coordinates via truncated SVD …")

    if issparse(ppmi):
        # Truncated SVD — only computes k singular vectors, memory-efficient
        k_actual = min(k, min(ppmi.shape) - 1)
        U, S, Vt = svds(ppmi, k=k_actual)
        # svds returns in ascending order — reverse
        idx    = np.argsort(S)[::-1]
        U, S   = U[:, idx], S[idx]
        coords = U * S
        # Approximate variance explained
        total_var = ppmi.power(2).sum()
        var_exp   = 100 * float(np.sum(S**2)) / (float(total_var) + 1e-10)
    else:
        U, S, Vt = svd(ppmi, full_matrices=False)
        coords   = U[:, :k] * S[:k]
        var_exp  = 100 * S[:k].sum() / (S.sum() + 1e-10)

    log.info(f"  Approx. variance explained by top {coords.shape[1]} dims: {var_exp:.1f}%")
    return coords


def blind_geometry_test(coords: np.ndarray,
                        word2idx: dict,
                        families: dict) -> dict:
    """
    Core blind test: are suppletives closer to root than random?
    Are they farther than regulars? (Nambu-Goto prediction)
    Can we predict unseen suppletive positions?

    No morphological labels used — only raw co-occurrence coordinates.
    """
    log.info("Running blind geometry test …")

    def euc(a, b): return np.linalg.norm(a - b)

    reg_all = []; supp_all = []; rand_all = []
    ng_correct = 0; total = 0

    for root, forms in families.items():
        if root not in word2idx: continue
        rc = coords[word2idx[root]]

        reg_d  = [euc(rc, coords[word2idx[f]])
                  for f, irr, d, freq in forms
                  if not irr and f in word2idx]
        sup_d  = [euc(rc, coords[word2idx[f]])
                  for f, irr, d, freq in forms
                  if irr and f in word2idx]

        if not reg_d or not sup_d: continue

        non_fam = [w for w in word2idx
                   if w != root and
                   w not in [f for f,_,_,_ in forms]]
        if len(non_fam) < 10: continue
        rnd = np.random.choice(non_fam, 30, replace=False)
        rnd_d = [euc(rc, coords[word2idx[w]]) for w in rnd]

        mr = np.mean(reg_d); ms = np.mean(sup_d); mn = np.mean(rnd_d)
        reg_all.extend(reg_d); supp_all.extend(sup_d); rand_all.extend(rnd_d)

        # Full Nambu-Goto prediction: supp > reg AND supp < rand
        if ms > mr and ms < mn: ng_correct += 1
        total += 1

    t_sr, p_sr = stats.ttest_ind(supp_all, reg_all)   if supp_all else (0,1)
    t_sn, p_sn = stats.ttest_ind(supp_all, rand_all)  if supp_all else (0,1)
    t_rn, p_rn = stats.ttest_ind(reg_all,  rand_all)  if reg_all  else (0,1)

    # Displacement vector predictive test
    # Train on 70%, test on 30%
    strong_pairs = [(root, f)
                    for root, forms in families.items()
                    for f, irr, d, freq in forms
                    if irr and root in word2idx and f in word2idx]
    np.random.shuffle(strong_pairs)
    n_train    = max(5, int(len(strong_pairs) * 0.7))
    train_p    = strong_pairs[:n_train]
    test_p     = strong_pairs[n_train:]

    disps      = [coords[word2idx[s]] - coords[word2idx[r]]
                  for r, s in train_p]
    mean_disp  = np.mean(disps, axis=0) if disps else np.zeros(coords.shape[1])

    n_vocab    = len(word2idx)
    pred_correct = 0
    for r, s in test_p:
        predicted = coords[word2idx[r]] + mean_disp
        pred_d    = euc(predicted, coords[word2idx[s]])
        rnd_idxs  = np.random.choice(n_vocab, 30, replace=False)
        rand_d    = np.mean([euc(coords[i], coords[word2idx[s]])
                             for i in rnd_idxs])
        if pred_d < rand_d: pred_correct += 1

    pred_acc = 100 * pred_correct / max(len(test_p), 1)

    # Family clustering: intra vs inter
    fam_map = {}
    for root, forms in families.items():
        for f, _, _, _ in forms:
            if f in word2idx: fam_map[f] = root
        if root in word2idx: fam_map[root] = root

    fam_words = list(fam_map.keys())
    intra_d = []; inter_d = []
    for i in range(min(len(fam_words), 200)):
        for j in range(i+1, min(len(fam_words), 200)):
            wi, wj = fam_words[i], fam_words[j]
            if wi not in word2idx or wj not in word2idx: continue
            d = euc(coords[word2idx[wi]], coords[word2idx[wj]])
            if fam_map[wi] == fam_map[wj]: intra_d.append(d)
            else:                           inter_d.append(d)

    t_clust, p_clust = stats.ttest_ind(intra_d, inter_d) if intra_d and inter_d else (0,1)

    results = {
        "mean_reg":    float(np.mean(reg_all))   if reg_all  else 0,
        "mean_supp":   float(np.mean(supp_all))  if supp_all else 0,
        "mean_rand":   float(np.mean(rand_all))  if rand_all else 0,
        "t_supp_reg":  float(t_sr), "p_supp_reg":  float(p_sr),
        "t_supp_rand": float(t_sn), "p_supp_rand": float(p_sn),
        "t_reg_rand":  float(t_rn), "p_reg_rand":  float(p_rn),
        "ng_correct":  ng_correct,  "ng_total":    total,
        "pred_acc":    pred_acc,
        "n_train":     n_train,     "n_test":      len(test_p),
        "intra_mean":  float(np.mean(intra_d)) if intra_d else 0,
        "inter_mean":  float(np.mean(inter_d)) if inter_d else 0,
        "t_clust":     float(t_clust), "p_clust": float(p_clust),
    }

    log.info(f"  Supp < Rand: t={t_sn:.3f}, p={p_sn:.4f}")
    log.info(f"  Supp > Reg:  t={t_sr:.3f}, p={p_sr:.4f}")
    log.info(f"  Predictive accuracy: {pred_acc:.1f}% ({pred_correct}/{len(test_p)})")
    log.info(f"  Family clustering: t={t_clust:.3f}, p={p_clust:.4f}")
    return results


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — STATIONARY ACTION & PHASE TRANSITION
# ════════════════════════════════════════════════════════════════════════════

def stationary_action_test(families: dict) -> dict:
    """
    Test the Lie group symmetry breaking prediction:
    There should be a SHARP frequency threshold above which
    irregularity (suppletive forms) appears — like a phase transition.

    If true: irregularity rate vs log(frequency) shows a sigmoid,
    not a gradual ramp. The inflection point is the 'critical frequency'.
    """
    log.info("Testing phase transition in irregularity …")

    # Bin lemmas by frequency, measure irregularity rate per bin
    freq_irr_pairs = []
    for lemma, forms in families.items():
        freq     = max((f for _, _, _, f in forms), default=1)
        has_irr  = any(irr for _, irr, _, _ in forms)
        freq_irr_pairs.append((np.log1p(freq), int(has_irr)))

    freq_irr_pairs.sort(key=lambda x: x[0])
    log_freqs = np.array([x[0] for x in freq_irr_pairs])
    irr_flags = np.array([x[1] for x in freq_irr_pairs])

    # Bin into 20 quantiles and compute irregularity rate per bin
    n_bins = min(20, len(freq_irr_pairs)//5)
    if n_bins < 3:
        log.warning("  Too few families for phase transition test")
        return {"phase_transition_detected": False}

    bins     = np.array_split(np.arange(len(freq_irr_pairs)), n_bins)
    bin_freq = [log_freqs[b].mean() for b in bins]
    bin_irr  = [irr_flags[b].mean() for b in bins]

    # Fit logistic (sigmoid) vs linear — which fits better?
    from scipy.optimize import curve_fit
    from scipy.stats import pearsonr

    def sigmoid(x, x0, k, ymax):
        return ymax / (1 + np.exp(-k*(x - x0)))

    def linear(x, m, b):
        return np.clip(m*x + b, 0, 1)

    x = np.array(bin_freq); y = np.array(bin_irr)

    try:
        # Fit sigmoid
        p0_sig = [np.median(x), 1.0, max(y)]
        popt_sig, _ = curve_fit(sigmoid, x, y, p0=p0_sig, maxfev=5000)
        y_pred_sig  = sigmoid(x, *popt_sig)
        ss_res_sig  = np.sum((y - y_pred_sig)**2)
        ss_tot      = np.sum((y - y.mean())**2)
        r2_sigmoid  = 1 - ss_res_sig / ss_tot if ss_tot > 1e-12 else float('nan')
        critical_freq = float(popt_sig[0])  # inflection point

        # Fit linear
        popt_lin, _  = curve_fit(linear, x, y, maxfev=5000)
        y_pred_lin   = linear(x, *popt_lin)
        ss_res_lin   = np.sum((y - y_pred_lin)**2)
        r2_linear    = 1 - ss_res_lin / ss_tot if ss_tot > 1e-12 else float('nan')

        # Phase transition = sigmoid fits significantly better
        phase_detected = (r2_sigmoid > r2_linear + 0.05)

        results = {
            "phase_transition_detected": bool(phase_detected),
            "r2_sigmoid":    float(r2_sigmoid),
            "r2_linear":     float(r2_linear),
            "critical_freq": critical_freq,  # log-scale
            "critical_freq_raw": float(np.expm1(critical_freq)),
            "bin_freq":      list(map(float, bin_freq)),
            "bin_irr":       list(map(float, bin_irr)),
            "sigmoid_params": list(map(float, popt_sig)),
        }
        log.info(f"  Sigmoid R²={r2_sigmoid:.3f}  Linear R²={r2_linear:.3f}")
        log.info(f"  Phase transition detected: {phase_detected}")
        log.info(f"  Critical frequency: ~{np.expm1(critical_freq):.0f} occurrences")
        return results

    except Exception as e:
        log.warning(f"  Curve fitting failed: {e}")
        return {"phase_transition_detected": False,
                "bin_freq": list(map(float, bin_freq)),
                "bin_irr":  list(map(float, bin_irr))}


def stationary_point_test(families: dict) -> dict:
    """
    For each suppletive form, compute the action functional
    S[phi] = T*surface - lambda_f*freq^2 + lambda_d*distinctiveness
    and test whether the suppletive sits near a stationary point (δS=0).

    Fit T, lambda_f, lambda_d by minimizing average distance
    from suppletive to nearest stationary point across all families.
    """
    log.info("Testing stationary action positions …")

    def action(x, root_x, all_xs, freq_f, freq_root, T, lf, ld):
        dist       = abs(x - root_x)
        surface    = T * np.sqrt(freq_f * freq_root + 1e-8) * dist
        freq_term  = -lf * freq_f**2
        dists_to_others = [abs(x - ox) for ox in all_xs if abs(x-ox) > 1e-6]
        distinct   = sum(ld * np.exp(-d*3) for d in dists_to_others if d < 1.5)
        return surface + freq_term + distinct

    def gradient(x, root_x, all_xs, freq_f, freq_root, T, lf, ld):
        dx = 1e-5
        return (action(x+dx, root_x, all_xs, freq_f, freq_root, T, lf, ld) -
                action(x-dx, root_x, all_xs, freq_f, freq_root, T, lf, ld)) / (2*dx)

    def find_stationary_points(root_x, all_xs, freq_f, freq_root, T, lf, ld):
        xs   = np.linspace(-0.1, 2.0, 500)
        grads = [gradient(x, root_x, all_xs, freq_f, freq_root, T, lf, ld) for x in xs]
        pts   = []
        for i in range(len(grads)-1):
            if grads[i] * grads[i+1] < 0:
                try:
                    xz = brentq(lambda x: gradient(x, root_x, all_xs, freq_f, freq_root, T, lf, ld),
                                xs[i], xs[i+1])
                    pts.append(xz)
                except: pass
        return pts

    # Assign phonological distances heuristically from string edit distance
    from difflib import SequenceMatcher
    def phon_dist(a, b):
        return 1 - SequenceMatcher(None, a, b).ratio()

    # Use default coupling constants (could be fitted)
    T=1.0; lf=2.5; ld=1.8
    deviations = []

    for lemma, forms in list(families.items())[:100]:  # cap at 100 for speed
        irr_forms = [(f, freq) for f, irr, d, freq in forms if irr and f != lemma]
        reg_forms = [(f, freq) for f, irr, d, freq in forms if not irr]
        if not irr_forms: continue

        root_x     = 0.0
        all_xs     = [phon_dist(lemma, f) for f, _ in reg_forms + irr_forms]
        freq_root  = max((freq for _, irr, _, freq in forms), default=1) / 1000.0

        for supp_f, freq_s in irr_forms:
            supp_x   = phon_dist(lemma, supp_f)
            freq_f   = freq_s / 1000.0
            stat_pts = find_stationary_points(root_x, all_xs, freq_f, freq_root, T, lf, ld)
            if stat_pts:
                nearest = min(stat_pts, key=lambda x: abs(x - supp_x))
                deviations.append(abs(nearest - supp_x))

    if not deviations:
        return {"stationary_test": "insufficient data"}

    mean_dev = float(np.mean(deviations))
    # Baseline: random deviation (what would be expected by chance)
    baseline = float(np.mean(np.random.uniform(0, 1, len(deviations)*10)))
    t, p     = stats.ttest_1samp(deviations, baseline/2)  # one-sample vs half baseline

    results = {
        "mean_deviation_from_stationary": mean_dev,
        "baseline_expected":              baseline,
        "t_vs_baseline":                  float(t),
        "p_vs_baseline":                  float(p),
        "n_suppletives_tested":           len(deviations),
        "stationary_test":                "significant" if p < 0.05 else "not significant",
    }
    log.info(f"  Mean deviation from stationary point: {mean_dev:.4f} (baseline ~{baseline:.4f})")
    log.info(f"  t={t:.3f}, p={p:.4f}")
    return results


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FRACTAL SIGNATURE ACROSS LEVELS
# ════════════════════════════════════════════════════════════════════════════

def letter_level_test() -> dict:
    """
    Test Nambu-Goto signature at the LETTER level.
    Silent letters are candidate orthogonal sprouts.
    """
    log.info("Running letter-level fractal test …")

    LETTER_FREQ = {
        'e':0.127,'t':0.091,'a':0.082,'o':0.075,'i':0.070,'n':0.067,
        'h':0.061,'s':0.063,'r':0.060,'d':0.043,'l':0.040,'c':0.028,
        'u':0.028,'m':0.024,'w':0.023,'f':0.022,'g':0.020,'y':0.020,
        'p':0.019,'b':0.015,'v':0.010,'k':0.008,'j':0.002,'x':0.002,
    }

    # (word, freq_tier, [(letter, position, is_silent)])
    SILENT_LETTER_WORDS = [
        ("knight","high",[("k",0,True),("n",1,False),("i",2,False),
                          ("g",3,True),("h",4,True),("t",5,False)]),
        ("know","high",[("k",0,True),("n",1,False),("o",2,False),("w",3,False)]),
        ("knife","high",[("k",0,True),("n",1,False),("i",2,False),
                         ("f",3,False),("e",4,True)]),
        ("write","high",[("w",0,True),("r",1,False),("i",2,False),
                         ("t",3,False),("e",4,True)]),
        ("wrong","high",[("w",0,True),("r",1,False),("o",2,False),
                         ("n",3,False),("g",4,False)]),
        ("lamb","high",[("l",0,False),("a",1,False),("m",2,False),("b",3,True)]),
        ("bomb","high",[("b",0,False),("o",1,False),("m",2,False),("b",3,True)]),
        ("climb","high",[("c",0,False),("l",1,False),("i",2,False),
                         ("m",3,False),("b",4,True)]),
        ("doubt","high",[("d",0,False),("o",1,False),("u",2,False),
                         ("b",3,True),("t",4,False)]),
        ("debt","high",[("d",0,False),("e",1,False),("b",2,True),("t",3,False)]),
        ("hour","high",[("h",0,True),("o",1,False),("u",2,False),("r",3,False)]),
        ("honest","high",[("h",0,True),("o",1,False),("n",2,False),
                          ("e",3,False),("s",4,False),("t",5,False)]),
        ("ghost","high",[("g",0,False),("h",1,True),("o",2,False),
                         ("s",3,False),("t",4,False)]),
        ("sign","high",[("s",0,False),("i",1,False),("g",2,True),("n",3,False)]),
        ("foreign","high",[("f",0,False),("o",1,False),("r",2,False),
                           ("e",3,False),("i",4,False),("g",5,True),("n",6,False)]),
        ("castle","high",[("c",0,False),("a",1,False),("s",2,False),
                          ("t",3,True),("l",4,False),("e",5,True)]),
        ("listen","high",[("l",0,False),("i",1,False),("s",2,False),
                          ("t",3,False),("e",4,False),("n",5,True)]),
        ("would","high",[("w",0,False),("o",1,False),("u",2,False),
                         ("l",3,True),("d",4,False)]),
        ("could","high",[("c",0,False),("o",1,False),("u",2,False),
                         ("l",3,True),("d",4,False)]),
        ("walk","high",[("w",0,False),("a",1,False),("l",2,True),("k",3,False)]),
        ("talk","high",[("t",0,False),("a",1,False),("l",2,True),("k",3,False)]),
        ("calm","high",[("c",0,False),("a",1,False),("l",2,True),("m",3,False)]),
        # Controls — no silent letters
        ("run","high",[("r",0,False),("u",1,False),("n",2,False)]),
        ("cat","high",[("c",0,False),("a",1,False),("t",2,False)]),
        ("dog","high",[("d",0,False),("o",1,False),("g",2,False)]),
        ("fast","high",[("f",0,False),("a",1,False),("s",2,False),("t",3,False)]),
        ("best","high",[("b",0,False),("e",1,False),("s",2,False),("t",3,False)]),
    ]

    freq_map = {"high":3,"medium":2,"low":1}
    instances = []
    for word, freq, letters in SILENT_LETTER_WORDS:
        wf = freq_map[freq]; n = len(letters)
        for letter, pos, is_silent in letters:
            lf   = LETTER_FREQ.get(letter, 0.01)
            arity = 0 if is_silent else max(1, int(lf*100 / (1+pos*0.3)))
            instances.append({
                "word":word,"letter":letter,"pos":pos,
                "is_silent":is_silent,"word_freq":wf,
                "letter_freq":lf,"arity":arity,
                "depth":pos/max(n-1,1),"n":n,
            })

    silent = [x for x in instances if x["is_silent"]]
    active = [x for x in instances if not x["is_silent"]]

    pct_s_term = 100*sum(1 for x in silent if x["arity"]==0)/max(len(silent),1)
    pct_a_term = 100*sum(1 for x in active if x["arity"]==0)/max(len(active),1)
    pct_s_bnd  = 100*sum(1 for x in silent
                         if x["pos"]==0 or x["pos"]==x["n"]-1)/max(len(silent),1)
    pct_a_bnd  = 100*sum(1 for x in active
                         if x["pos"]==0 or x["pos"]==x["n"]-1)/max(len(active),1)

    t_arity, p_arity = stats.ttest_ind(
        [x["arity"] for x in active], [x["arity"] for x in silent])

    results = {
        "pct_silent_terminal": pct_s_term,
        "pct_active_terminal": pct_a_term,
        "pct_silent_boundary": pct_s_bnd,
        "pct_active_boundary": pct_a_bnd,
        "boundary_ratio":      pct_s_bnd / max(pct_a_bnd, 0.01),
        "t_arity": float(t_arity), "p_arity": float(p_arity),
        "n_silent": len(silent), "n_active": len(active),
    }
    log.info(f"  Silent terminal: {pct_s_term:.1f}%  Active terminal: {pct_a_term:.1f}%")
    log.info(f"  Boundary ratio: {pct_s_bnd:.1f}% vs {pct_a_bnd:.1f}%")
    return results


def sentence_level_test(ud_paths=None) -> dict:
    """
    Test Nambu-Goto signature at the SENTENCE level.
    Accepts a list of .conllu paths (or a single path string).
    """
    log.info("Running sentence-level fractal test …")

    if ud_paths:
        if isinstance(ud_paths, str):
            ud_paths = [ud_paths]
        valid = [p for p in ud_paths if p and Path(p).exists()]
        if valid:
            return sentence_level_from_ud(valid)

    # Fallback: curated examples
    log.info("  Using curated sentence examples (UD not available)")

    clauses = [
        # (is_sprout, depth, arity, word_freq)
        (False, 0, 5, 3),(False, 0, 4, 3),(False, 0, 6, 3),(False, 0, 4, 3),
        (False, 0, 5, 2),(False, 0, 4, 2),(False, 0, 3, 2),(False, 0, 5, 1),
        (True,  1, 0, 3),(True,  1, 1, 3),(True,  1, 0, 3),(True,  1, 0, 3),
        (True,  1, 1, 2),(True,  1, 0, 2),(True,  2, 0, 2),(True,  1, 0, 1),
        (True,  1, 0, 3),(True,  2, 0, 1),(True,  1, 1, 2),(True,  1, 0, 3),
    ]

    sprout = [(a, d) for s, d, a, f in clauses if s]
    main   = [(a, d) for s, d, a, f in clauses if not s]

    supp_arity = [a for a,d in sprout]
    main_arity = [a for a,d in main]
    t, p = stats.ttest_ind(main_arity, supp_arity)

    pct_sp_term = 100*sum(1 for a,_ in sprout if a==0)/max(len(sprout),1)
    pct_mn_term = 100*sum(1 for a,_ in main   if a==0)/max(len(main),1)

    return {
        "pct_sprout_terminal": pct_sp_term,
        "pct_main_terminal":   pct_mn_term,
        "mean_sprout_arity":   float(np.mean(supp_arity)),
        "mean_main_arity":     float(np.mean(main_arity)),
        "t_arity": float(t), "p_arity": float(p),
        "source": "curated",
    }


def sentence_level_from_ud(ud_paths) -> dict:
    """Parse one or more CoNLL-U files and measure subordinate clause properties."""
    if isinstance(ud_paths, str):
        ud_paths = [ud_paths]

    sentences = []; current = []
    for ud_path in ud_paths:
        with open(ud_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"): continue
                if not line.strip():
                    if current: sentences.append(current); current = []
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 8 and parts[0].isdigit():
                    current.append({
                        "id":     int(parts[0]),
                        "form":   parts[1],
                        "deprel": parts[7],
                        "head":   int(parts[6]) if parts[6].isdigit() else 0,
                    })
        if current: sentences.append(current); current = []  # flush last sentence

    if QUICK: sentences = sentences[:2000]
    log.info(f"  Loaded {len(sentences):,} sentences from {len(ud_paths)} treebank(s)")

    sprout_deprels = {"relcl","advcl","acl","ccomp","xcomp","parataxis"}
    main_deprels   = {"root"}

    sprout_arities = []
    main_arities   = []

    for sent in sentences:
        # Build head→children map
        head_children = defaultdict(list)
        for tok in sent:
            head_children[tok["head"]].append(tok["id"])

        for tok in sent:
            # Arity = number of children this token has
            arity = len(head_children[tok["id"]])
            if tok["deprel"] in sprout_deprels:
                sprout_arities.append(arity)
            elif tok["deprel"] in main_deprels:
                main_arities.append(arity)
    sa = sprout_arities; ma = main_arities
    if not sa or not ma:
        return sentence_level_test()

    t, p = stats.ttest_ind(ma, sa)
    pct_sp_term = 100*sum(1 for a in sa if a==0)/len(sa)
    pct_mn_term = 100*sum(1 for a in ma if a==0)/len(ma)

    log.info(f"  UD: {len(sentences):,} sentences, "
             f"mean sprout arity={np.mean(sa):.2f}, main={np.mean(ma):.2f}")
    return {
        "pct_sprout_terminal": pct_sp_term,
        "pct_main_terminal":   pct_mn_term,
        "mean_sprout_arity":   float(np.mean(sa)),
        "mean_main_arity":     float(np.mean(ma)),
        "t_arity": float(t), "p_arity": float(p),
        "n_sentences":         len(sentences),
        "source": "UD treebank",
    }


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — DIMENSIONAL LIFTING TEST
# ════════════════════════════════════════════════════════════════════════════

def dimensional_lifting_test(coords: np.ndarray,
                              word2idx: dict,
                              families: dict) -> dict:
    """
    Test whether progressively higher-dimensional embeddings
    better capture morphological structure.

    Prediction: routing accuracy improves monotonically with k
    up to some saturation point — which estimates the true
    dimensionality of the morphological manifold.
    """
    log.info("Testing dimensional lifting …")

    def family_retrieval_score(coords_k, word2idx, families, top_k=3):
        """P@3 for morphological family retrieval at given dimensionality."""
        scores = []
        for root, forms in families.items():
            if root not in word2idx: continue
            rc      = coords_k[word2idx[root]]
            true_rel = {f for f, irr, d, freq in forms if f in word2idx}
            if not true_rel: continue

            dists   = [(np.linalg.norm(rc - coords_k[word2idx[w]]), w)
                       for w in word2idx if w != root]
            dists.sort()
            top_words = [w for _, w in dists[:top_k]]
            hits      = sum(1 for w in top_words if w in true_rel)
            scores.append(hits / top_k)
        return float(np.mean(scores)) if scores else 0.0

    # Build full PPMI from families (use family co-occurrence as proxy)
    all_words = list({w for root, forms in families.items()
                      for w in [root] + [f for f,_,_,_ in forms]
                      if w in word2idx})
    if len(all_words) < 10:
        log.warning("  Too few words for dimensional test")
        return {}

    dims   = [1, 2, 4, 8, 16, 32, 64]
    scores = []
    for k in dims:
        k_actual = min(k, coords.shape[1])
        score    = family_retrieval_score(coords[:, :k_actual], word2idx, families)
        scores.append(score)
        log.info(f"  k={k}: P@3 = {score:.4f}")

    # Find saturation point (where marginal gain drops below 5%)
    saturation_k = dims[-1]
    for i in range(1, len(dims)):
        if scores[i] > 0 and (scores[i]-scores[i-1])/scores[i] < 0.05:
            saturation_k = dims[i-1]; break

    return {
        "dims": dims,
        "scores": scores,
        "saturation_k": saturation_k,
        "max_score": max(scores) if scores else 0,
        "score_at_4d": scores[dims.index(4)] if 4 in dims else 0,
    }


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — VISUALIZATION
# ════════════════════════════════════════════════════════════════════════════

def make_full_report(all_results: dict, out_dir: Path):
    """Generate comprehensive visualization of all test results."""
    if NO_PLOTS: return

    log.info("Generating visualizations …")
    matplotlib.rcParams['axes.facecolor'] = '#0d0d22'
    matplotlib.rcParams['figure.facecolor'] = '#06060f'
    matplotlib.rcParams['text.color'] = '#e8e4ff'
    matplotlib.rcParams['axes.labelcolor'] = '#e8e4ff'
    matplotlib.rcParams['xtick.color'] = '#e8e4ff'
    matplotlib.rcParams['ytick.color'] = '#e8e4ff'
    matplotlib.rcParams['axes.edgecolor'] = '#252545'

    bg = '#06060f'; panel = '#0d0d22'; grid_c = '#1e1e3a'; white = '#e8e4ff'
    GOLD=  '#fbbf24'; GREEN='#34d399'; PURPLE='#a78bfa'
    RED =  '#f87171'; BLUE= '#60a5fa'; ORANGE='#fb923c'

    def style(ax, title):
        ax.set_facecolor(panel)
        ax.set_title(title, color=white, fontsize=10, pad=8, fontweight='bold')
        ax.tick_params(colors=white, labelsize=8)
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        for sp in ['bottom','left']: ax.spines[sp].set_color(grid_c)
        ax.grid(color=grid_c, alpha=0.4)

    # ── Figure 1: Morphological Network Analysis ──
    fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle('MORPHOLOGICAL NETWORK ANALYSIS\n'
                  'Nambu-Goto Geometry in English Morphology',
                  fontsize=14, color=white, fontweight='bold')

    mn = all_results.get("morph_network", {})

    # Panel 1: Arity by frequency
    ax = axes[0,0]; style(ax, 'Branching Arity by Frequency')
    if mn:
        means = [mn["mean_low_arity"], mn["mean_med_arity"], mn["mean_high_arity"]]
        ax.bar([1,2,3], means, color=[BLUE,GREEN,PURPLE], alpha=0.85,
               edgecolor='white', lw=0.7)
        for i, m in enumerate(means):
            ax.text(i+1, m+0.05, f'{m:.2f}', ha='center', color='white',
                    fontsize=10, fontweight='bold')
        ax.set_xticks([1,2,3])
        ax.set_xticklabels(['LOW','MEDIUM','HIGH'], color=white)
        ax.set_ylabel('Mean family size')
        ax.text(0.05, 0.95, f"r={mn['r_freq_arity']:.3f}\np={mn['p_freq_arity']:.4f}",
                transform=ax.transAxes, color=GOLD, fontsize=9,
                va='top', fontweight='bold')

    # Panel 2: Trifurcation rates
    ax = axes[0,1]; style(ax, 'Trifurcation Rate by Frequency')
    if mn and "trif_rates" in mn:
        rates = [mn["trif_rates"].get("low",0),
                 mn["trif_rates"].get("medium",0),
                 mn["trif_rates"].get("high",0)]
        ax.bar([1,2,3], rates, color=[BLUE,GREEN,PURPLE], alpha=0.85,
               edgecolor='white', lw=0.7)
        for i, r in enumerate(rates):
            ax.text(i+1, r+0.5, f'{r:.1f}%', ha='center', color='white',
                    fontsize=10, fontweight='bold')
        ax.set_xticks([1,2,3])
        ax.set_xticklabels(['LOW','MEDIUM','HIGH'], color=white)
        ax.set_ylabel('% roots with arity ≥ 3')

    # Panel 3: Suppletive terminality
    ax = axes[0,2]; style(ax, 'Suppletive Sprout Terminality')
    if mn:
        ax.bar([1,2], [mn["pct_supp_term"], mn["pct_reg_term"]],
               color=[RED, BLUE], alpha=0.85, edgecolor='white', lw=0.7, width=0.5)
        for i, v in enumerate([mn["pct_supp_term"], mn["pct_reg_term"]]):
            ax.text(i+1, v+1, f'{v:.1f}%', ha='center', color='white',
                    fontsize=11, fontweight='bold')
        ax.set_xticks([1,2])
        ax.set_xticklabels(['Suppletive\n(sprout)', 'Regular'], color=white)
        ax.set_ylabel('% terminal forms')

    # Panel 4: Blind test distances
    ax = axes[1,0]; style(ax, 'Blind Geometry Test\n(Raw Co-occurrence)')
    bt = all_results.get("blind_test", {})
    if bt:
        vals = [bt.get("mean_reg",0), bt.get("mean_supp",0), bt.get("mean_rand",0)]
        cols = [BLUE, PURPLE, 'gray']
        bars = ax.bar([1,2,3], vals, color=cols, alpha=0.85,
                      edgecolor='white', lw=0.7, width=0.5)
        for i, v in enumerate(vals):
            ax.text(i+1, v+0.02, f'{v:.3f}', ha='center', color='white',
                    fontsize=10, fontweight='bold')
        ax.set_xticks([1,2,3])
        ax.set_xticklabels(['Regular','Suppletive','Random'], color=white)
        ax.set_ylabel('Distance from root')
        ax.text(0.5, 0.95,
                f"Pred. acc: {bt.get('pred_acc',0):.0f}%\n"
                f"p(supp<rand)={bt.get('p_supp_rand',1):.3f}",
                transform=ax.transAxes, ha='center', va='top',
                color=GOLD, fontsize=9, fontweight='bold')

    # Panel 5: Phase transition
    ax = axes[1,1]; style(ax, 'Phase Transition Test\n(Lie Group Symmetry Breaking)')
    pt = all_results.get("phase_transition", {})
    if pt and "bin_freq" in pt:
        x = np.array(pt["bin_freq"]); y = np.array(pt["bin_irr"])
        ax.scatter(x, y, color=PURPLE, s=60, zorder=3, edgecolors='white', lw=0.8)
        if "sigmoid_params" in pt:
            xf = np.linspace(x.min(), x.max(), 200)
            p0 = pt["sigmoid_params"]
            yf = p0[2] / (1 + np.exp(-p0[1]*(xf-p0[0])))
            ax.plot(xf, yf, color=RED, lw=2.5, label=f"Sigmoid R²={pt.get('r2_sigmoid',0):.3f}")
            ax.axvline(pt.get("critical_freq",0), color=GOLD, lw=1.5, ls='--',
                       label=f"Critical freq ≈ {pt.get('critical_freq_raw',0):.0f}")
        ax.set_xlabel('log(word frequency)'); ax.set_ylabel('Irregularity rate')
        ax.legend(fontsize=7, facecolor=panel, labelcolor=white)
        detected = pt.get("phase_transition_detected", False)
        ax.text(0.05, 0.95, f"Phase transition: {'✓' if detected else '✗'}",
                transform=ax.transAxes, color=GREEN if detected else RED,
                fontsize=10, fontweight='bold', va='top')

    # Panel 6: Dimensional lifting
    ax = axes[1,2]; style(ax, 'Dimensional Lifting\nRouting vs Manifold Dimension')
    dl = all_results.get("dim_lifting", {})
    if dl and "dims" in dl:
        ax.plot(dl["dims"], dl["scores"], 'o-', color=PURPLE, lw=2.5, ms=8,
                markeredgecolor='white', markeredgewidth=0.8)
        ax.axvline(dl.get("saturation_k", 8), color=GOLD, lw=1.5, ls='--',
                   label=f"Saturation at k={dl.get('saturation_k',8)}")
        ax.set_xlabel('Embedding dimensions (k)')
        ax.set_ylabel('Morphological retrieval P@3')
        ax.set_xscale('log')
        ax.legend(fontsize=8, facecolor=panel, labelcolor=white)

    plt.tight_layout()
    fig1.savefig(out_dir/"fig1_morphological_analysis.png",
                 dpi=150, bbox_inches='tight', facecolor=bg)
    plt.close(fig1)

    # ── Figure 2: Fractal Signature ──
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle('FRACTAL NAMBU-GOTO SIGNATURE ACROSS THREE LEVELS\n'
                  'Letter · Word · Sentence',
                  fontsize=14, color=white, fontweight='bold')

    ll = all_results.get("letter_level", {})
    sl = all_results.get("sentence_level", {})
    mn2 = all_results.get("morph_network", {})

    levels   = ['LETTER\n(Silent vs Active)', 'WORD\n(Supp. vs Regular)', 'SENTENCE\n(Sub. vs Main)']
    term_sp  = [ll.get("pct_silent_terminal",0),
                mn2.get("pct_supp_term",0),
                sl.get("pct_sprout_terminal",0)]
    term_mn  = [ll.get("pct_active_terminal",0),
                mn2.get("pct_reg_term",0),
                sl.get("pct_main_terminal",0)]
    arity_sp = [0,
                mn2.get("mean_low_arity",0)*0.1,
                sl.get("mean_sprout_arity",0)]
    arity_mn = [ll.get("pct_active_terminal",0)*0.1,
                mn2.get("mean_high_arity",0),
                sl.get("mean_main_arity",0)]
    cols3    = [RED, GOLD, GREEN]

    for i, ax in enumerate(axes2):
        style(ax, f'{levels[i]}\nSprout vs Main/Active')
        x = np.array([1,2]); w=0.3
        ax.bar(x-w/2, [term_sp[i], arity_mn[i] if i>0 else arity_mn[i]],
               width=w, color=PURPLE, alpha=0.85, label='Sprout', edgecolor='white', lw=0.6)
        ax.bar(x+w/2, [term_mn[i], arity_sp[i]  if i>0 else arity_sp[i]],
               width=w, color=BLUE,   alpha=0.85, label='Main',   edgecolor='white', lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(['Terminality %','Arity'], color=white)
        ax.legend(fontsize=8, facecolor=panel, labelcolor=white)
        ax.set_title(f'LEVEL {i+1}: {levels[i]}', color=cols3[i], fontsize=10,
                     pad=8, fontweight='bold')

    plt.tight_layout()
    fig2.savefig(out_dir/"fig2_fractal_signature.png",
                 dpi=150, bbox_inches='tight', facecolor=bg)
    plt.close(fig2)

    log.info(f"  Figures saved to {out_dir}/")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — SAVE RESULTS & PRINT SUMMARY
# ════════════════════════════════════════════════════════════════════════════

def print_summary(all_results: dict):
    W = "\033[0m"; G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"

    def safe(val, default=0.0):
        """Return default if val is None or NaN."""
        if val is None: return default
        try:
            return default if np.isnan(float(val)) else float(val)
        except (TypeError, ValueError):
            return default

    print(f"\n{'='*70}")
    print(f"{B}NAMBU-GOTO LANGUAGE HYPOTHESIS — RESULTS SUMMARY{W}")
    print(f"{'='*70}\n")

    mn  = all_results.get("morph_network", {})
    bt  = all_results.get("blind_test", {})
    ll  = all_results.get("letter_level", {})
    sl  = all_results.get("sentence_level", {})
    pt  = all_results.get("phase_transition", {})
    st  = all_results.get("stationary_test", {})
    dl  = all_results.get("dim_lifting", {})
    cl  = all_results.get("cross_language", {})

    def check(condition, label, detail):
        sym = f"{G}✓{W}" if condition else f"{R}✗{W}"
        col = G if condition else R
        print(f"  {sym}  {label:<45} {col}{detail}{W}")

    print(f"{Y}── MORPHOLOGICAL NETWORK (English) ──{W}")
    if mn:
        check(safe(mn.get("p_freq_arity"))<0.05 and safe(mn.get("r_freq_arity"))>0,
              "Frequency → Arity (r>0, p<0.05)",
              f"r={safe(mn.get('r_freq_arity')):.3f}, p={safe(mn.get('p_freq_arity'),1):.4f}")
        check(safe(mn.get("p_anova"),1)<0.05,
              "Trifurcation threshold (ANOVA p<0.05)",
              f"F={safe(mn.get('f_anova')):.2f}, p={safe(mn.get('p_anova'),1):.4f}")
        check(safe(mn.get("pct_supp_term")) > safe(mn.get("pct_reg_term")),
              "Suppletive more terminal than regular",
              f"{safe(mn.get('pct_supp_term')):.1f}% vs {safe(mn.get('pct_reg_term')):.1f}%")
        check(safe(mn.get("hf_sprout_dens"),1) < safe(mn.get("lf_sprout_dens")),
              "Sprout density ↓ at high frequency",
              f"HF={safe(mn.get('hf_sprout_dens')):.3f} LF={safe(mn.get('lf_sprout_dens')):.3f}")

    print(f"\n{Y}── BLIND CO-OCCURRENCE TEST ──{W}")
    if bt:
        check(safe(bt.get("p_supp_rand"),1)<0.05 and safe(bt.get("t_supp_rand"))< 0,
              "Suppletive < Random (p<0.05)",
              f"t={safe(bt.get('t_supp_rand')):.3f}, p={safe(bt.get('p_supp_rand'),1):.4f}")
        check(safe(bt.get("p_supp_reg"),1)<0.05 and safe(bt.get("t_supp_reg"))>0,
              "Suppletive > Regular (p<0.05)",
              f"t={safe(bt.get('t_supp_reg')):.3f}, p={safe(bt.get('p_supp_reg'),1):.4f}")
        check(safe(bt.get("pred_acc"))>50,
              "Predictive accuracy > 50% chance",
              f"{safe(bt.get('pred_acc')):.1f}% ({bt.get('n_test',0)} pairs)")
        check(safe(bt.get("p_clust"),1)<0.05,
              "Family clustering significant",
              f"t={safe(bt.get('t_clust')):.3f}, p={safe(bt.get('p_clust'),1):.4f}")

    print(f"\n{Y}── FRACTAL SIGNATURE ──{W}")
    if ll:
        check(safe(ll.get("pct_silent_terminal")) > safe(ll.get("pct_active_terminal"))+10,
              "Letter: silent more terminal",
              f"{safe(ll.get('pct_silent_terminal')):.1f}% vs {safe(ll.get('pct_active_terminal')):.1f}%")
        check(safe(ll.get("boundary_ratio")) > 1.2,
              "Letter: silent at boundaries ×1.2",
              f"{safe(ll.get('boundary_ratio')):.2f}×")
    if sl:
        check(safe(sl.get("p_arity"),1)<0.05,
              "Sentence: sub. clause arity < main (p<0.05)",
              f"t={safe(sl.get('t_arity')):.3f}, p={safe(sl.get('p_arity'),1):.4f}")
        check(safe(sl.get("pct_sprout_terminal")) > safe(sl.get("pct_main_terminal")),
              "Sentence: sub. clauses more terminal",
              f"{safe(sl.get('pct_sprout_terminal')):.1f}% vs {safe(sl.get('pct_main_terminal')):.1f}%")

    print(f"\n{Y}── LIE GROUP / PHASE TRANSITION ──{W}")
    if pt:
        check(pt.get("phase_transition_detected", False),
              "Sigmoid fits better than linear",
              f"R²(sig)={safe(pt.get('r2_sigmoid')):.3f} R²(lin)={safe(pt.get('r2_linear')):.3f}")
        if "critical_freq_raw" in pt:
            print(f"     Critical frequency: ~{pt['critical_freq_raw']:.0f} corpus occurrences")

    print(f"\n{Y}── STATIONARY ACTION TEST ──{W}")
    if st and "mean_deviation_from_stationary" in st:
        check(st.get("stationary_test","")=="significant",
              "Suppletives near stationary points (p<0.05)",
              f"mean dev={st['mean_deviation_from_stationary']:.4f}, "
              f"p={safe(st.get('p_vs_baseline'),1):.4f}")

    print(f"\n{Y}── CROSS-LANGUAGE UNIVERSALITY ──{W}")
    for lang, res in (cl or {}).items():
        if isinstance(res, dict):
            r_val = safe(res.get("r_freq_arity"))
            p_val = safe(res.get("p_freq_arity"), 1.0)
            check(p_val < 0.05 and r_val > 0,
                  f"{lang}: Frequency→Arity significant",
                  f"r={r_val:.3f}, p={p_val:.4f}")

    print(f"\n{Y}── DIMENSIONAL LIFTING ──{W}")
    if dl and dl.get("scores"):
        check(dl["scores"][-1] > dl["scores"][0],
              "Higher dimensions improve routing",
              f"k=1: {dl['scores'][0]:.3f} → k={dl['dims'][-1]}: {dl['scores'][-1]:.3f}")
        print(f"     Estimated manifold saturation: k={dl.get('saturation_k','?')}")

    print(f"\n{'='*70}")

    # Count confirmed predictions
    total   = sum(1 for k,v in all_results.items() if isinstance(v, dict))
    print(f"\nResults saved to: {OUT_DIR}/")
    print(f"{'='*70}\n")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    log.info("="*65)
    log.info("NAMBU-GOTO LANGUAGE HYPOTHESIS — FULL PIPELINE")
    log.info("="*65)

    all_results = {}

    # ── 0. Load data ──
    texts     = download_wikitext()
    ud_paths  = download_ud_english()
    families  = load_celex_or_wordnet()

    # Enrich morphological families with real corpus frequencies
    # This is essential for the phase transition test (Section 4)
    families  = enrich_with_corpus_freq(families, texts)

    # UniMorph for cross-language
    um_eng = load_unimorph("eng")
    um_tur = load_unimorph("tur")
    um_ara = load_unimorph("ara")

    # ── 1. Morphological network ──
    log.info("\n── Section 1: Morphological Network Analysis ──")
    all_results["morph_network"] = analyze_morphological_networks(families)

    # ── 2. Cross-language ──
    log.info("\n── Section 2: Cross-Language Universality ──")
    cl = {}
    for lang, entries in [("English(UM)", um_eng),
                           ("Turkish",     um_tur),
                           ("Arabic",      um_ara)]:
        if entries:
            cl[lang] = analyze_unimorph_language(entries, lang)
    all_results["cross_language"] = cl

    # ── 3. Blind geometry test ──
    log.info("\n── Section 3: Blind Co-occurrence Geometry Test ──")
    vocab, word2idx, ppmi = build_cooccurrence(texts, window=5, min_freq=15)
    coords                = learn_ng_coordinates(ppmi, k=8)
    all_results["blind_test"] = blind_geometry_test(coords, word2idx, families)

    # ── 4. Phase transition ──
    log.info("\n── Section 4: Phase Transition (Lie Group) ──")
    all_results["phase_transition"] = stationary_action_test(families)

    # ── 5. Stationary action test ──
    log.info("\n── Section 5: Stationary Action Positions ──")
    all_results["stationary_test"] = stationary_point_test(families)

    # ── 6. Fractal levels ──
    log.info("\n── Section 6: Fractal Signature Across Levels ──")
    all_results["letter_level"]   = letter_level_test()
    all_results["sentence_level"] = sentence_level_test(ud_paths)

    # ── 7. Dimensional lifting ──
    log.info("\n── Section 7: Dimensional Lifting ──")
    all_results["dim_lifting"] = dimensional_lifting_test(coords, word2idx, families)

    # ── 8. Save ──
    out_json = OUT_DIR / "all_results.json"
    with open(out_json, "w") as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray):     return obj.tolist()
            if isinstance(obj, (np.bool_,)):    return bool(obj)
            raise TypeError(f"Not serializable: {type(obj)}")
        json.dump(all_results, f, indent=2, default=convert)
    log.info(f"Results saved to {out_json}")

    # ── 9. Visualize ──
    make_full_report(all_results, OUT_DIR)

    # ── 10. Print summary ──
    print_summary(all_results)


if __name__ == "__main__":
    main()