
"""
CENG442 - Assignment 1
Azerbaijani text preprocessing + Word2Vec / FastText training
Saves two-column Excel files, corpus_all.txt, embeddings/word2vec.model, embeddings/fasttext.model
"""

import re
import html
import unicodedata
from pathlib import Path
import argparse
import logging
import random
import sys

import pandas as pd
from ftfy import fix_text

from gensim.models import Word2Vec, FastText
import numpy as np

# ----------------------------
# Config / Logging / Seed
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
RND_SEED = 42
random.seed(RND_SEED)
np.random.seed(RND_SEED)

# ----------------------------
# Regex & helpers (from spec)
# ----------------------------
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE      = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE    = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)
PHONE_RE    = re.compile(r"\+?\d[\d\-\s\(\)]{6,}\d")
USER_RE     = re.compile(r"@\w+")
MULTI_PUNCT = re.compile(r"([!?.,;:])\1{1,}")
MULTI_SPACE = re.compile(r"\s+")
REPEAT_CHARS= re.compile(r"(.)\1{2,}", flags=re.UNICODE)

TOKEN_RE = re.compile(
    r"[A-Za-zƏəĞğIıİiÖöÜüÇçŞşXxQq]+(?:'[A-Za-zƏəĞğIıİiÖöÜüÇçŞşXxQq]+)?"
    r"|<NUM>|URL|EMAIL|PHONE|USER|EMO_(?:POS|NEG)"
)

EMO_MAP = {"🙂":"EMO_POS","😀":"EMO_POS","😍":"EMO_POS","😊":"EMO_POS","👍":"EMO_POS",
           "☹":"EMO_NEG","🙁":"EMO_NEG","😠":"EMO_NEG","😡":"EMO_NEG","👎":"EMO_NEG"}

SLANG_MAP = {"slm":"salam","tmm":"tamam","sagol":"sağol","cox":"çox","yaxsi":"yaxşı"}
NEGATORS  = {"yox","deyil","heç","qətiyyən","yoxdur"}

# Domain hints (from spec)
NEWS_HINTS   = re.compile(r"\b(apa|trend|azertac|reuters|bloomberg|dha|aa)\b", re.I)
SOCIAL_HINTS = re.compile(r"\b(rt)\b|@|#|(?:😂|😍|😊|👍|👎|😡|🙂)")
REV_HINTS    = re.compile(r"\b(azn|manat|qiymət|aldım|ulduz|çox yaxşı|çox pis)\b", re.I)

PRICE_RE     = re.compile(r"\b\d+\s*(azn|manat)\b", re.I)
STARS_RE     = re.compile(r"\b([1-5])\s*ulduz\b", re.I)
POS_RATE     = re.compile(r"\bçox yaxşı\b")
NEG_RATE     = re.compile(r"\bçox pis\b")

# ----------------------------
# Azerbaijani-aware lowercase
# ----------------------------
def lower_az(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFC", s)
    # Replace dotted/dotless I properly before lowercasing
    s = s.replace("I", "ı").replace("İ", "i")
    s = s.lower()
    # fix possible combining dot on i
    s = s.replace("i\u0307", "i")
    return s

# ----------------------------
# Domain detection & domain-specific normalization
# ----------------------------
def detect_domain(text: str) -> str:
    if not isinstance(text, str):
        return "general"
    s = text.lower()
    if NEWS_HINTS.search(s):
        return "news"
    if SOCIAL_HINTS.search(s):
        return "social"
    if REV_HINTS.search(s):
        return "reviews"
    return "general"

def domain_specific_normalize(cleaned: str, domain: str) -> str:
    if domain == "reviews":
        s = PRICE_RE.sub(" <PRICE> ", cleaned)
        s = STARS_RE.sub(lambda m: f" <STARS_{m.group(1)}> ", s)
        s = POS_RATE.sub(" <RATING_POS> ", s)
        s = NEG_RATE.sub(" <RATING_NEG> ", s)
        return " ".join(s.split())
    return cleaned

def add_domain_tag(line: str, domain: str) -> str:
    return f"dom{domain} " + line  # e.g., 'domnews ...'

# ----------------------------
# Normalization / Cleaning
# ----------------------------
def normalize_text_az(s: str, numbers_to_token=True, keep_sentence_punct=False) -> str:
    if not isinstance(s, str): return ""
    # Emoji mapping first
    for emo, tag in EMO_MAP.items():
        s = s.replace(emo, f" {tag} ")
    s = fix_text(s)
    s = html.unescape(s)
    s = HTML_TAG_RE.sub(" ", s)
    s = URL_RE.sub(" URL ", s)
    s = EMAIL_RE.sub(" EMAIL ", s)
    s = PHONE_RE.sub(" PHONE ", s)
    # Hashtag: keep text, split camelCase
    s = re.sub(r"#([A-Za-z0-9_]+)",
               lambda m: " " + re.sub('([a-z])([A-Z])', r'\1 \2', m.group(1)) + " ", s)
    s = USER_RE.sub(" USER ", s)
    s = lower_az(s)
    s = MULTI_PUNCT.sub(r"\1", s)
    if numbers_to_token:
        s = re.sub(r"\d+", " <NUM> ", s)
    if keep_sentence_punct:
        s = re.sub(r"[^\w\s<>'əğıöşüçƏĞIİÖŞÜÇxqXQ.!?]", " ", s)
    else:
        s = re.sub(r"[^\w\s<>'əğıöşüçƏĞIİÖŞÜÇxqXQ]", " ", s)
    s = MULTI_SPACE.sub(" ", s).strip()
    toks = TOKEN_RE.findall(s)
    norm = []
    mark_neg = 0
    for t in toks:
        t = REPEAT_CHARS.sub(r"\1\1", t)  # collapse repeats to max 2
        t = SLANG_MAP.get(t, t)
        if t in NEGATORS:
            norm.append(t)
            mark_neg = 3
            continue
        if mark_neg > 0 and t not in {"URL","EMAIL","PHONE","USER"}:
            norm.append(t + "_NEG")
            mark_neg -= 1
        else:
            norm.append(t)
    # Remove single-letter tokens except 'o' and 'e'
    norm = [t for t in norm if not (len(t) == 1 and t not in {"o","e"})]
    return " ".join(norm).strip()

# ----------------------------
# Mapping sentiment values
# ----------------------------
def map_sentiment_value(v, scheme: str):
    if scheme == "binary":
        try:
            # Accept numeric 0/1 or strings '0','1'
            if isinstance(v, (int, float)):
                return 1.0 if int(v) == 1 else 0.0
            v_s = str(v).strip()
            return 1.0 if int(v_s) == 1 else 0.0
        except Exception:
            return None
    s = str(v).strip().lower()
    if s in {"pos","positive","1","müsbət","mussebet","müsbət","good","pozitiv","müsbət"}:
        return 1.0
    if s in {"neu","neutral","2","neytral","neytral"}:
        return 0.5
    if s in {"neg","negative","0","mənfi","menfi","bad","neqativ"}:
        return 0.0
    # Try some Azerbaijani words explicitly:
    if s in {"müsbət","müsbətlik","müsbətlikli"}:
        return 1.0
    if s in {"neytral","neytrallıq"}:
        return 0.5
    if s in {"mənfi","pis"}:
        return 0.0
    return None

# ----------------------------
# Process a single file and save two-column Excel
# ----------------------------
def process_file(in_path, text_col, label_col, scheme, out_two_col_path, remove_stopwords=False):
    logging.info(f"Processing: {in_path} -> {out_two_col_path} (scheme={scheme})")
    df = pd.read_excel(in_path, engine="openpyxl")
    # Drop common accidental columns
    for c in ["Unnamed: 0","index"]:
        if c in df.columns:
            df = df.drop(columns=[c])
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Missing expected columns in {in_path}: need '{text_col}' and '{label_col}'")
    # Keep only non-empty text
    df = df.dropna(subset=[text_col])
    df = df[df[text_col].astype(str).str.strip().str.len() > 0]
    # Drop duplicates by raw text
    df = df.drop_duplicates(subset=[text_col])

    # Base cleaning
    df["cleaned_text"] = df[text_col].astype(str).apply(lambda s: normalize_text_az(s))
    # Domain detection from original raw text (for domain-specific normalization)
    df["__domain__"] = df[text_col].astype(str).apply(detect_domain)
    df["cleaned_text"] = df.apply(lambda r: domain_specific_normalize(r["cleaned_text"], r["__domain__"]), axis=1)

    # Optional stopword removal (kept minimal)
    if remove_stopwords:
        sw = set(["və","ilə","amma","ancaq","lakin","ya","həm","ki","bu","bir","o","biz","siz","mən","sən",
                  "orada","burada","bütün","hər","artıq","çox","az","ən","də","da","üçün"])
        for keep in ["deyil","yox","heç","qətiyyən","yoxdur"]:
            sw.discard(keep)
        df["cleaned_text"] = df["cleaned_text"].apply(lambda s: " ".join([t for t in s.split() if t not in sw]))

    # sentiment mapping
    df["sentiment_value"] = df[label_col].apply(lambda v: map_sentiment_value(v, scheme))
    df = df.dropna(subset=["sentiment_value"])
    df["sentiment_value"] = df["sentiment_value"].astype(float)

    out_df = df[["cleaned_text","sentiment_value"]].reset_index(drop=True)
    Path(out_two_col_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_excel(out_two_col_path, index=False, engine="openpyxl")
    logging.info(f"Saved: {out_two_col_path} (rows={len(out_df)})")
    return out_two_col_path

# ----------------------------
# Build combined corpus_all.txt
# ----------------------------
def build_corpus_txt(input_files, text_cols, out_txt="corpus_all.txt"):
    
    logging.info("Building corpus_all.txt ...")
    lines = []
    for (f, text_col) in zip(input_files, text_cols):
        df = pd.read_excel(f, engine="openpyxl")
        print(f"\nDomain counts in {f}:")
        print(df[text_col].astype(str).apply(detect_domain).value_counts())
        for raw in df[text_col].dropna().astype(str):
            dom = detect_domain(raw)
            s = normalize_text_az(raw, keep_sentence_punct=True)
            parts = re.split(r"[.!?]+", s)
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                # Remove remaining punctuation allowed set to Azerbaijani letters and tokens
                p = re.sub(r"[^\w\səğıöşüçƏĞIİÖŞÜÇxqXQ<>\-]+", " ", p)
                p = " ".join(p.split()).lower()
                if p:
                    lines.append(add_domain_tag(p, dom))
    with open(out_txt, "w", encoding="utf-8") as w:
        for ln in lines:
            w.write(ln + "\n")
    logging.info(f"Wrote {out_txt} with {len(lines)} lines")
    return out_txt

# ----------------------------
# Train embeddings
# ----------------------------
def train_embeddings(two_col_files, w2v_out="embeddings/word2vec.model", ft_out="embeddings/fasttext.model",
                     vector_size=300, window=5, min_count=3, epochs=10, sg=1):
    logging.info("Loading sentences from two-column files ...")
    sentences = []
    for f in two_col_files:
        df = pd.read_excel(f, usecols=["cleaned_text"], engine="openpyxl")
        sentences.extend(df["cleaned_text"].astype(str).str.split().tolist())
    # gensim expects list of token lists
    logging.info(f"Total sentences for training: {len(sentences)}")
    Path("embeddings").mkdir(exist_ok=True)
    # Word2Vec
    logging.info("Training Word2Vec ...")
    w2v = Word2Vec(sentences=sentences, vector_size=vector_size, window=window, min_count=min_count,
                   sg=sg, negative=10, epochs=epochs, seed=RND_SEED)
    w2v.save(w2v_out)
    logging.info(f"Saved Word2Vec model -> {w2v_out}")
    # FastText
    logging.info("Training FastText ...")
    ft = FastText(sentences=sentences, vector_size=vector_size, window=window, min_count=min_count,
                  sg=sg, min_n=3, max_n=6, epochs=epochs, seed=RND_SEED)
    ft.save(ft_out)
    logging.info(f"Saved FastText model -> {ft_out}")
    return w2v_out, ft_out

# ----------------------------
# Basic comparisons / reports
# ----------------------------
def lexical_coverage(model, tokens):
    vocab = model.wv.key_to_index
    return sum(1 for t in tokens if t in vocab) / max(1, len(tokens))

def read_tokens_from_two_col(fpath):
    df = pd.read_excel(fpath, usecols=["cleaned_text"], engine="openpyxl")
    return [t for row in df["cleaned_text"].astype(str) for t in row.split()]

def compare_models(w2v_path, ft_path, two_col_files):
    logging.info("Comparing models ...")
    w2v = Word2Vec.load(w2v_path)
    ft = FastText.load(ft_path)
    # sample metrics
    files = two_col_files
    for f in files:
        toks = read_tokens_from_two_col(f)
        cov_w2v = lexical_coverage(w2v, toks)
        cov_ftv = lexical_coverage(ft, toks)
        logging.info(f"{Path(f).name}: W2V coverage={cov_w2v:.3f}, FT vocab coverage={cov_ftv:.3f}")

    seed_words = ["yaxşı","pis","çox","bahalı","ucuz","mükəmməl","dəhşət","<PRICE>","<RATING_POS>"]
    syn_pairs  = [("yaxşı","əla"), ("bahalı","qiymətli"), ("ucuz","sərfəli")]
    ant_pairs  = [("yaxşı","pis"), ("bahalı","ucuz")]

    def pair_sim(model, pairs):
        vals = []
        for a,b in pairs:
            try:
                vals.append(model.wv.similarity(a,b))
            except KeyError:
                pass
        return sum(vals)/len(vals) if vals else float('nan')

    syn_w2v = pair_sim(w2v, syn_pairs)
    syn_ft  = pair_sim(ft,  syn_pairs)
    ant_w2v = pair_sim(w2v, ant_pairs)
    ant_ft  = pair_sim(ft,  ant_pairs)
    logging.info(f"Synonyms: W2V={syn_w2v:.3f}, FT={syn_ft:.3f}")
    logging.info(f"Antonyms: W2V={ant_w2v:.3f}, FT={ant_ft:.3f}")
    # nearest neighbors
    def neighbors(model, word, k=5):
        try:
            return [w for w,_ in model.wv.most_similar(word, topn=k)]
        except Exception:
            return []
    for w in seed_words:
        logging.info(f"W2V NN for '{w}': {neighbors(w2v, w)}")
        logging.info(f"FT  NN for '{w}': {neighbors(ft, w)}")

# ----------------------------
# Main CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="CENG442 - Azerbaijani preprocessing & embeddings")
    parser.add_argument("--in-dir", type=str, default=".", help="Directory containing source xlsx files")
    parser.add_argument("--out-dir", type=str, default=".", help="Directory to save outputs")
    parser.add_argument("--train-embeddings", action="store_true", help="Train Word2Vec + FastText after preprocess")
    parser.add_argument("--epochs", type=int, default=10, help="Embedding training epochs")
    parser.add_argument("--min-count", type=int, default=3, help="min_count for embeddings")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CFG from assignment
    CFG = [
        ("labeled-sentiment.xlsx",        "text", "sentiment", "tri"),
        ("test__1_.xlsx",                 "text", "label",     "binary"),
        ("train__3_.xlsx",                "text", "label",     "binary"),
        ("train-00000-of-00001.xlsx",     "text", "labels",    "tri"),
        ("merged_dataset_CSV__1_.xlsx",   "text", "labels",    "binary"),
    ]

    two_col_files = []
    for fname, tcol, lcol, scheme in CFG:
        in_path = in_dir / fname
        if not in_path.exists():
            logging.warning(f"Input file not found: {in_path} — skipping")
            continue
        out_name = f"{Path(fname).stem}_2col.xlsx"
        out_path = out_dir / out_name
        try:
            p = process_file(in_path, tcol, lcol, scheme, out_path, remove_stopwords=False)
            two_col_files.append(str(p))
        except Exception as e:
            logging.exception(f"Failed processing {in_path}: {e}")

    if len(two_col_files) == 0:
        logging.error("No files processed — exiting.")
        sys.exit(1)

    # Build corpus_all.txt using the original files (not two-col) to preserve domain detection
    input_files = [in_dir / c[0] for c in CFG if (in_dir / c[0]).exists()]
    build_corpus_txt([str(p) for p in input_files], [c[1] for c in CFG if (in_dir / c[0]).exists()], out_txt=str(out_dir / "corpus_all.txt"))

    if args.train_embeddings:
        w2v_path, ft_path = train_embeddings(two_col_files, w2v_out=str(out_dir / "embeddings/word2vec.model"),
                                            ft_out=str(out_dir / "embeddings/fasttext.model"),
                                            vector_size=300, window=5, min_count=args.min_count, epochs=args.epochs, sg=1)
        # Basic comparison
        compare_models(w2v_path, ft_path, two_col_files)
    else:
        logging.info("Skipping embedding training. Use --train-embeddings to train models.")

if __name__ == "__main__":
    main()
