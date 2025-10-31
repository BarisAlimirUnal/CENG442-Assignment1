# pipeline.py
# CENG442 – End-to-end pipeline: 2col + corpus + (opt) w2v/ft + comparison

import re, html, unicodedata, argparse, logging, random, sys
from pathlib import Path
import pandas as pd
import numpy as np
try:
    from ftfy import fix_text
except Exception:
    def fix_text(s): return s
try:
    from gensim.models import Word2Vec, FastText
except Exception:
    Word2Vec = FastText = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
RND_SEED = 42
random.seed(RND_SEED); np.random.seed(RND_SEED)

# ---------- Regex & helpers (PDF tarzı) ----------
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE      = re.compile(r"(https?://\S+|www\.\S+)", re.I)
EMAIL_RE    = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.I)
PHONE_RE    = re.compile(r"\+?\d[\d\-\s\(\)]{6,}\d")
USER_RE     = re.compile(r"@\w+")
MULTI_PUNCT = re.compile(r"([!?.,;:])\1{1,}")
MULTI_SPACE = re.compile(r"\s+")
REPEAT_CHARS= re.compile(r"(.)\1{2,}", re.UNICODE)
TOKEN_RE    = re.compile(r"[A-Za-zƏəĞğIıİiÖöÜüÇçŞşXxQq]+(?:'[A-Za-zƏəĞğIıİiÖöÜüÇçŞşXxQq]+)?|<NUM>|URL|EMAIL|PHONE|USER|EMO_(?:POS|NEG)")
EMO_MAP     = {"🙂":"EMO_POS","😀":"EMO_POS","😍":"EMO_POS","😊":"EMO_POS","👍":"EMO_POS",
               "☹":"EMO_NEG","🙁":"EMO_NEG","😠":"EMO_NEG","😡":"EMO_NEG","👎":"EMO_NEG","😭":"EMO_NEG"}
SLANG_MAP   = {"slm":"salam","tmm":"tamam","sagol":"sağol","cox":"çox","yaxsi":"yaxşı"}
NEGATORS    = {"yox","deyil","heç","qətiyyən","yoxdur"}

# Domain hints (kısa, dengeli)
NEWS_HINTS = re.compile(
    r"\b(apa|trend|azertac|reuters|bloomberg|bbc|report|haqqin|day\.az|minval|aa|dha|xeber|xəbər|agentliyi)\b",
    re.I
)

SOCIAL_HINTS = re.compile(
    r"@|#|\b(rt|salam|allah|baxın|qaqa|qaqaş|vauu|superrr|ayy|😂|😍|😊|👍|👎|😡|🙂|😭|❤️|💔|🔥|💯)\b"
    r"|(?:x\.com|twitter\.com|t\.me|telegram\.me|instagram\.com|facebook\.com|tiktok\.com|youtube\.com|youtu\.be)",
    re.I
)

REV_HINTS = re.compile(
    r"\b(azn|manat|qiymət|qiymet|aldım|aldim|sifariş|sifaris|ulduz|məhsul|mehsul|satıcı|satici|endirim|kampaniya"
    r"|bahalı|bahali|ucuz|çox yaxşı|cox yaxsi|çox pis|cox pis|mükəmməl|super|yaxşı|pis)\b",
    re.I
)

PRICE_RE     = re.compile(r"\b\d+\s*(azn|manat)\b", re.I)
STARS_RE     = re.compile(r"\b([1-5])\s*ulduz\b", re.I)
POS_RATE     = re.compile(r"\bçox yaxşı\b")
NEG_RATE     = re.compile(r"\bçox pis\b")

def lower_az(s:str)->str:
    if not isinstance(s,str): return ""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("I","ı").replace("İ","i").lower().replace("i\u0307","i")
    return s

def detect_domain(text:str)->str:
    s = text.lower()
    if NEWS_HINTS.search(s): return "news"
    if SOCIAL_HINTS.search(s): return "social"
    if REV_HINTS.search(s):   return "reviews"
    return "general"

def domain_specific_normalize(cleaned:str, dom:str)->str:
    if dom=="reviews":
        s = PRICE_RE.sub(" <PRICE> ", cleaned)
        s = STARS_RE.sub(lambda m: f" <STARS_{m.group(1)}> ", s)
        s = POS_RATE.sub(" <RATING_POS> ", s)
        s = NEG_RATE.sub(" <RATING_NEG> ", s)
        return " ".join(s.split())
    return cleaned

def normalize_text_az(s:str, numbers_to_token=True, keep_sentence_punct=False)->str:
    if not isinstance(s,str): return ""
    for emo,tag in EMO_MAP.items(): s = s.replace(emo,f" {tag} ")
    s = fix_text(s); s = html.unescape(s)
    s = HTML_TAG_RE.sub(" ", s)
    s = URL_RE.sub(" URL ", s)
    s = EMAIL_RE.sub(" EMAIL ", s)
    s = PHONE_RE.sub(" PHONE ", s)
    s = re.sub(r"#([A-Za-z0-9_]+)", lambda m: " "+re.sub('([a-z])([A-Z])', r'\1 \2', m.group(1))+" ", s)
    s = USER_RE.sub(" USER ", s)
    s = lower_az(s)
    s = MULTI_PUNCT.sub(r"\1", s)
    if numbers_to_token: s = re.sub(r"\d+"," <NUM> ", s)
    s = re.sub(r"[^\w\s<>'əğıöşüçƏĞIİÖŞÜÇxqXQ.!?]" if keep_sentence_punct else r"[^\w\s<>'əğıöşüçƏĞIİÖŞÜÇxqXQ]"," ", s)
    s = MULTI_SPACE.sub(" ", s).strip()
    toks = TOKEN_RE.findall(s)
    out, neg = [], 0
    for t in toks:
        t = REPEAT_CHARS.sub(r"\1\1", t)
        t = SLANG_MAP.get(t,t)
        if t in NEGATORS:
            out.append(t); neg=3; continue
        if neg>0 and t not in {"URL","EMAIL","PHONE","USER"}:
            out.append(t+"_NEG"); neg-=1
        else:
            out.append(t)
    out = [t for t in out if not (len(t)==1 and t not in {"o","e"})]
    return " ".join(out).strip()

def map_sentiment_value(v, scheme:str):
    if scheme=="binary":
        try: return 1.0 if int(str(v).strip())==1 else 0.0
        except Exception: return None
    s = str(v).strip().lower()
    if s in {"pos","positive","1","müsbət","good","pozitiv"}: return 1.0
    if s in {"neu","neutral","2","neytral"}: return 0.5
    if s in {"neg","negative","0","mənfi","bad","neqativ"}: return 0.0
    return None

# ---------- Preprocess ----------
def process_file(in_path, text_col, label_col, scheme, out_two_col_path, remove_stopwords=False):
    df = pd.read_excel(in_path, engine="openpyxl")
    for c in ("Unnamed: 0","index"):
        if c in df.columns: df = df.drop(columns=[c])
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Missing columns in {in_path}: need {text_col}, {label_col}")
    df = df.dropna(subset=[text_col])
    df = df[df[text_col].astype(str).str.strip().str.len()>0]
    df = df.drop_duplicates(subset=[text_col])

    df["cleaned_text"] = df[text_col].astype(str).apply(normalize_text_az)
    df["__domain__"]   = df[text_col].astype(str).apply(detect_domain)
    df["cleaned_text"] = df.apply(lambda r: domain_specific_normalize(r["cleaned_text"], r["__domain__"]), axis=1)

    if remove_stopwords:
        sw = set(["və","ilə","amma","ancaq","lakin","ya","həm","ki","bu","bir","o","biz","siz","mən","sən","orada","burada","bütün","hər","artıq","çox","az","ən","də","da","üçün"])
        for keep in ["deyil","yox","heç","qətiyyən","yoxdur"]: sw.discard(keep)
        df["cleaned_text"] = df["cleaned_text"].apply(lambda s: " ".join(t for t in s.split() if t not in sw))

    df["sentiment_value"] = df[label_col].apply(lambda v: map_sentiment_value(v, scheme))
    df = df.dropna(subset=["sentiment_value"])
    df["sentiment_value"] = df["sentiment_value"].astype(float)

    out_df = df[["cleaned_text","sentiment_value"]].reset_index(drop=True)
    out_two_col_path = Path(out_two_col_path).resolve()
    out_two_col_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_excel(out_two_col_path, index=False, engine="openpyxl")
    logging.info(f"[OK] 2col -> {out_two_col_path} (rows={len(out_df)})")
    return str(out_two_col_path)

def build_corpus_txt(input_files, text_cols, out_txt):
    out_txt = Path(out_txt).resolve()
    lines = []
    for f, tcol in zip(input_files, text_cols):
        df = pd.read_excel(f, engine="openpyxl")
        for raw in df[tcol].dropna().astype(str):
            dom = detect_domain(raw)
            s = normalize_text_az(raw, keep_sentence_punct=True)
            parts = re.split(r"[.!?]+", s)
            for p in parts:
                p = p.strip()
                if not p: continue
                p = re.sub(r"[^\w\səğıöşüçƏĞIİÖŞÜÇxqXQ]"," ", p)
                p = " ".join(p.split()).lower()
                if p: lines.append(f"dom{dom} {p}")
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as w:
        for ln in lines: w.write(ln+"\n")
    logging.info(f"[OK] corpus -> {out_txt} (lines={len(lines)})")
    return str(out_txt)

# ---------- Train (optional) ----------
def train_embeddings(two_col_files, w2v_out, ft_out, vector_size=300, window=5, min_count=3, epochs=10, sg=1):
    if Word2Vec is None or FastText is None:
        raise RuntimeError("gensim bulunamadı. `pip install gensim`")
    sentences = []
    for f in two_col_files:
        df = pd.read_excel(f, usecols=["cleaned_text"], engine="openpyxl")
        sentences.extend(df["cleaned_text"].astype(str).str.split().tolist())
    Path(w2v_out).parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"[EMB] sentences={len(sentences)}")
    w2v = Word2Vec(sentences=sentences, vector_size=vector_size, window=window, min_count=min_count, sg=sg, negative=10, epochs=epochs, seed=RND_SEED)
    w2v.save(w2v_out); logging.info(f"[EMB] W2V -> {w2v_out}")
    ft  = FastText(sentences=sentences, vector_size=vector_size, window=window, min_count=min_count, sg=sg, min_n=3, max_n=6, epochs=epochs, seed=RND_SEED)
    ft.save(ft_out); logging.info(f"[EMB] FT  -> {ft_out}")
    return w2v_out, ft_out

# ---------- Compare (optional) ----------
def lexical_coverage(model, tokens):
    vocab = model.wv.key_to_index
    return sum(1 for t in tokens if t in vocab) / max(1, len(tokens))

def read_tokens_from_two_col(fpath):
    df = pd.read_excel(fpath, usecols=["cleaned_text"], engine="openpyxl")
    return [t for row in df["cleaned_text"].astype(str) for t in row.split()]

def compare_models(w2v_path, ft_path, two_col_files):
    if Word2Vec is None or FastText is None:
        raise RuntimeError("gensim bulunamadı.")
    w2v = Word2Vec.load(w2v_path); ft = FastText.load(ft_path)
    logging.info("[CMP] Lexical coverage per dataset:")
    for f in two_col_files:
        toks = read_tokens_from_two_col(f)
        cov_w2v = lexical_coverage(w2v, toks)
        cov_ftv = lexical_coverage(ft, toks)
        logging.info(f"  {Path(f).name}: W2V={cov_w2v:.3f}, FT(vocab)={cov_ftv:.3f}")
    seed_words = ["yaxşı","pis","çox","bahalı","ucuz","mükəmməl","dəhşət","<PRICE>","<RATING_POS>"]
    syn_pairs  = [("yaxşı","əla"), ("bahalı","qiymətli"), ("ucuz","sərfəli")]
    ant_pairs  = [("yaxşı","pis"), ("bahalı","ucuz")]
    def pair_sim(model,pairs):
        vals=[]; 
        for a,b in pairs:
            try: vals.append(model.wv.similarity(a,b))
            except KeyError: pass
        return sum(vals)/len(vals) if vals else float('nan')
    syn_w2v = pair_sim(w2v,syn_pairs); syn_ft = pair_sim(ft,syn_pairs)
    ant_w2v = pair_sim(w2v,ant_pairs); ant_ft = pair_sim(ft,ant_pairs)
    logging.info(f"[CMP] Syn: W2V={syn_w2v:.3f}, FT={syn_ft:.3f}")
    logging.info(f"[CMP] Ant: W2V={ant_w2v:.3f}, FT={ant_ft:.3f}")
    def neighbors(model, w, k=5):
        try: return [x for x,_ in model.wv.most_similar(w, topn=k)]
        except KeyError: return []
    for w in seed_words:
        logging.info(f"[CMP] W2V NN('{w}') -> {neighbors(w2v,w)}")
        logging.info(f"[CMP]  FT NN('{w}') -> {neighbors(ft, w)}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="CENG442 pipeline: preprocess + corpus + (opt) train + compare")
    ap.add_argument("--in-dir", type=str, default=".", help="Input dir with XLSX")
    ap.add_argument("--out-dir", type=str, default=".", help="Output dir")
    ap.add_argument("--remove-stopwords", action="store_true", help="Optional stopword removal")
    ap.add_argument("--train-embeddings", action="store_true", help="Train W2V + FT")
    ap.add_argument("--compare", action="store_true", help="Compare models after training")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--min-count", type=int, default=3)
    args = ap.parse_args()

    in_dir  = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"[CFG] in_dir={in_dir}")
    logging.info(f"[CFG] out_dir={out_dir}")

    CFG = [
        ("labeled-sentiment.xlsx",        "text", "sentiment", "tri"),
        ("test__1_.xlsx",                 "text", "label",     "binary"),
        ("train__3_.xlsx",                "text", "label",     "binary"),
        ("train-00000-of-00001.xlsx",     "text", "labels",    "tri"),
        ("merged_dataset_CSV__1_.xlsx",   "text", "labels",    "binary"),
    ]

    # 1) Preprocess → 2col
    two_col_files = []
    for fname, tcol, lcol, scheme in CFG:
        in_path = in_dir / fname
        if not in_path.exists():
            logging.warning(f"[SKIP] not found: {in_path}")
            continue
        out_path = (out_dir / f"{Path(fname).stem}_2col.xlsx").resolve()
        out_file = process_file(in_path, tcol, lcol, scheme, out_path, remove_stopwords=args.remove_stopwords)
        if not Path(out_file).exists():
            raise RuntimeError(f"2col not written: {out_file}")
        two_col_files.append(out_file)

    if not two_col_files:
        logging.error("No two-column outputs produced. Exiting.")
        sys.exit(1)

    # 2) Corpus
    input_files = [str((in_dir / c[0]).resolve()) for c in CFG if (in_dir / c[0]).exists()]
    text_cols   = [c[1] for c in CFG if (in_dir / c[0]).exists()]
    build_corpus_txt(input_files, text_cols, out_txt=str((out_dir / "corpus_all.txt").resolve()))

    # 3) Train (optional)
    if args.train_embeddings:
        w2v_out = str((out_dir / "embeddings/word2vec.model").resolve())
        ft_out  = str((out_dir / "embeddings/fasttext.model").resolve())
        train_embeddings(two_col_files, w2v_out, ft_out, vector_size=300, window=5, min_count=args.min_count, epochs=args.epochs, sg=1)
        # 4) Compare (optional)
        if args.compare:
            compare_models(w2v_out, ft_out, two_col_files)
    else:
        logging.info("[INFO] Skipping training. Use --train-embeddings to enable.")

if __name__ == "__main__":
    main()
