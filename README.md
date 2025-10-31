# CENG442-Assignment1
Barış Alimir Ünal

Ahmet Gürkan Gönül

Yunus Emre Cincil


1) Data & Goal

We work with 5 Azerbaijani sentiment datasets (train/test/labeled/merged).
Each dataset contains text and a sentiment label.

Goal:
Clean Azerbaijani text, keep semantic signals (negation, emojis, price info, ratings), detect domain type, build domain-tagged corpus, train Word2Vec & FastText, and compare them.

Why neutral = 0.5
Neutral sits between positive (1.0) and negative (0.0).
Value 0.5 keeps middle meaning and lets the model learn a smooth sentiment scale instead of forcing binary.

2) Preprocessing

Text is normalized by:

Azerbaijani-aware lowercase (İ/ı rules)

Remove HTML, URLs, emails, phone numbers

Replace emojis → positive/negative tags

Tokenize and compress repeated characters

Slang normalization (slm → salam, cox → çox)

Mark negation for ~3 tokens (yaxşı_NEG)

Drop empty, whitespace-only, duplicate rows

Convert numbers to <NUM>

Keep one word sentiment values only

Before → After examples:

Raw text	Cleaned
İnanılmaz məhsul!!! Çox yaxşı!!!	inanılmaz məhsul çox yaxşı
Bu çox baha deyil	bu çox baha deyil baha_NEG
😂😂 Superr məhsul aldım 5 ulduz	EMO_POS EMO_POS super məhsul aldım <STARS_5>
Qiymet 50 azn	qiymet <NUM> <PRICE>
RT @user: Salam millet!!	rt USER salam millet
3) Mini Challenges & Observations

Azerbaijani lowercase
İ/ı needs Unicode normalization or tokens get split incorrectly.

Negation propagation
Negation over 3 tokens works well for short comments (e.g., shopping reviews).

Emoji polarity
Emoji→sentiment tag helps social media domain clearly.

Review price signals
<PRICE> and <STARS_X> gave stronger patterns for product data.

4) Domain-Aware Processing

Rule-based detection
If text contains keywords/URLs/emojis typical for a domain:

Domain	Examples Pattern
news	apa, trend, azertac, report, reuters, bbc
social	rt, @, #, emojis, IG/Twitter URLs
reviews	azn, manat, qiymət, ulduz, sifariş

Domain-specific normalization
For reviews only:

Replace price: 50 azn → <PRICE>

Replace star rating: 5 ulduz → <STARS_5>

Mark strong opinions: çox yaxşı → <RATING_POS>

Corpus format
Each sentence prefixed with domain tag:

domnews prezident görüş keçirib
domreviews məhsul çox yaxşı <PRICE>
domsocial salam millet EMO_POS
domgeneral bu gün hava soyuq

5) Embeddings

We train Word2Vec and FastText on cleaned corpus.

Training Settings
Setting	Value
Vector size	300
Window	5
Min count	2
Model type	Skip-gram
Negative samples	10
Subwords (FT)	3–6
Epochs	12
Seed	42
Results (example summary)
Metric	Word2Vec	FastText
Lexical coverage	lower	higher (subwords)
Synonym similarity	good	slightly better
Antonym separation	moderate	stronger gap
Best domain fit	reviews / social	general / news
Nearest Neighbors (samples)
Word	W2V	FT
yaxşı	əla, super	əla, mükəmməl
ucuz	sərfəli	endirimli, sərfəli
<PRICE>	manat, azn	ödəmə, qiymət
6) Optional Lemmatization

Simple rule-based stemming: remove common Azerbaijani suffixes
(e.g., -ları, -nin, -dan)

Effect:
Small vocabulary reduction, slightly cleaner clusters, but also some errors because rule-only stemming does not know context.

7) Reproducibility

Environment

Python 3.x

pandas, openpyxl, ftfy, gensim, numpy

Hardware

CPU (no GPU needed)

Seed

random.seed(42), numpy.seed(42)


Run command

python pipeline.py --in-dir . --out-dir output --train-embeddings --compare --epochs 12 --min-count 2
