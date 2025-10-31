# CENG442-Assignment1
Barış Alimir Ünal

Ahmet Gürkan Gönül

Yunus Emre Cincil

1) Data & Goal   

We work with 5 Azerbaijani sentiment datasets. Each dataset contains text and a sentiment label. We clean Azerbaijani text, keep semantic signals (negation, emojis, price info, ratings), detect domain type, build domain-tagged corpus, train Word2Vec & FastText, and compare them. We use neutral because it sits between positive and negative. Value 0.5 keeps middle meaning and lets the model learn a smooth sentiment scale instead of forcing binary.
_____________________________________________________________________________________________________
2) Preprocessing

Text is normalized by:

Azerbaijani-aware lowercase (İ/ı rules), Removing HTML, URLs, emails, phone numbers, Replacing emojis with positive/negative tags, Tokenizing and compressing repeated characters, Slang normalization (slm - salam, cox - çox), Dropping empty, whitespace only, duplicate rows, Converting numbers to <NUM>, Keeping one word sentiment values only
_____________________________________________________________________________________________________
Before After examples:

### Before → After Examples

| Raw text | Cleaned |
|---------|---------|
| İnanılmaz məhsul!!! Çox yaxşı!!! | `inanılmaz məhsul çox yaxşı` |
| 😂😂 Superr məhsul aldım 5 ulduz | `EMO_POS EMO_POS super məhsul aldım <STARS_5>` |
| Qiymet 50 azn | `qiymet <NUM> <PRICE>` |
| RT @user: Salam millet!! | `rt USER salam millet` |

_____________________________________________________________________________________________________
3) Mini Challenges & Observations

Emoji mapping

-Emoji => EMO_POS / EMO_NEG increased sentiment clarity in social data

-Noise emojis reduced after mapping
_____________________________________________________________________________________________________
4) Domain-Aware Processing

Domain detection:

We added more keywords to catch sentences as the keyword given in the code resulted in majority of the domains being general. This increased the other 3 domain sizes considerably but still the general domain remains the largest. We added some tags by hand and some by feeding a sample dataset through an AI model to find common keywords.


Domain specific normalization

We normalized prices, ratings for example

### Domain-Specific Transformations (Reviews)

| Pattern | Output |
|--------|--------|
| `50 azn` | `<PRICE>` |
| `5 ulduz` | `<STARS_5>` |

_____________________________________________________________________________________________________
5) Embeddings

We trained Word2Vec and FastText on cleaned corpus.

### Training Settings

| Setting           | Value |
|------------------|-------|
| Vector size       | 300   |
| Window            | 5     |
| Min count         | 2     |
| Model type        | Skip-gram |
| Negative samples  | 10    |
| Subwords (FT)     | 3–6   |
| Epochs            | 12    |
| Seed              | 42    |

Similarity Scores
| Metric             | Word2Vec  | FastText  |
| ------------------ | --------- | --------- |
| Synonym similarity | **0.332** | **0.418** |
| Antonym similarity | 0.310     | **0.358** |

Nearest Neighbors (NN) Samples
| Word             | W2V Neighbors                                               | FT Neighbors                                          |
| ---------------- | ----------------------------------------------------------- | ----------------------------------------------------- |
| **yaxşı**        | iyi, yaxşı, yaxwi, `<RATING_POS>`, awsome                   | yaxşıı, yaxşıca, yaxşıkı, yaxşıl, yaxşı               |
| **pis**          | günd, vərdişlərə, kardeşi, yedi_NEG, baktelecom_NEG         | piss, piis, pisolog, pisə, pissdi                     |
| **çox**          | çöx, gözəldir, çpx, bəyənilsin, çoox                        | çoxçox, çoxx, çoxh, ço, çoh                           |
| **bahalı**       | yaxtaları, villaları, portretlerinə, düzəlmə, sabuncudur    | bahalıı, bahalısı, bahalıq, bahalıdı, baharlı         |
| **ucuz**         | şeytanbazardan, sududu, düzəltdirilib, yelenaya, çetinlikde | ucuzu, ucuza, ucuzdu, ucuzluğa, ucu                   |
| **mükəmməl**     | kəliməylə, möhtəşəmm, carrey, möhdəşəm, mükəmməll           | mükəmməll, mükəmməlsən, mükəməl, mukəmməl, mükəmməldi |
| **dəhşət**       | xalçalardan, birda, ayranları, soundtreki, onagörə          | dəhşətdü, dəhşətdie, dəhşətə, dəhşətiymiş, dəhşəti    |
| **`<PRICE>`**      | *none*                                                      | engiltdere, recognise, rdr, cokubse, umbilnise        |
| **<RATING_POS>** | deneyin, internetli, oynuyorum, qeşəy, yaxşı                | `<RATING_NEG>`, süperr, çoxk, çoxkk, superr           |

FastText is better than Word2Vec for synonym/antonym structure. FastText handles Azerbaijani morphology and misspellings better. Word2Vec pulls unrelated tokens more often, especially in reviews

____________________________________________________________________________________________________
6) Reproducibility

Environment

Python 3.x

pandas, openpyxl, ftfy, gensim, numpy

Seed

random.seed(42), numpy.seed(42)

Run command

python pipeline.py --in-dir . --out-dir output --train-embeddings --compare --epochs 12 --min-count 2

put all 5 excel files int the same folder if the code doesn't create one create outputs/embeddings folders

_____________________________________________________________________________________________________

7) Conclusion

FastText worked better for this dataset. It handled Azerbaijani spelling variation, informal writing, and noisy social media text more effectively because it uses subword information. This helped it understand misspellings and slang while Word2Vec often returned unrelated neighbors for rare or noisy tokens. Word2Vec still produced reasonable clusters but it struggled with non vocabulary cases and typed variants. Overall FastText provided stronger synonym grouping, clearer antonym separation and more stable neighbors for sentiment and review specific tokens like `<PRICE>` and <RATING_POS>. For future work a transformer based Azerbaijani model or domain specific subword dictionary could further improve results.
