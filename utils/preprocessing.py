import re
import pandas as pd
from datetime import datetime
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


# Stemmer & stopwords (Sastrawi)
_factory_stem = StemmerFactory()
_stemmer = _factory_stem.create_stemmer()
_stop_factory = StopWordRemoverFactory()
_STOPWORDS = set(_stop_factory.get_stop_words())


from datetime import datetime
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory




# Stemmer & stopwords (Sastrawi)
_factory_stem = StemmerFactory()
_stemmer = _factory_stem.create_stemmer()
_stop_factory = StopWordRemoverFactory()
_STOPWORDS = set(_stop_factory.get_stop_words())

def parse_date_safe(x):
    try:
        return pd.to_datetime(x, errors='coerce')
    except Exception:
        return pd.NaT

# Normalisasi slang dasar
BASE_NORMALIZER = {
    'gk':'tidak','ga':'tidak','nggak':'tidak','gak':'tidak','tdk':'tidak',
    'bgt':'banget','dr':'dari','utk':'untuk','dgn':'dengan','sm':'sama',
    'yg':'yang','sdh':'sudah','krn':'karena','tp':'tapi','dlm':'dalam',
    'sy':'saya','aku':'saya','gue':'saya','anda':'kamu','pd':'pada',
}

# === INI WAJIB ADA ===
URL_PATTERN       = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN   = re.compile(r"@[A-Za-z0-9_]+", re.IGNORECASE)
HASHTAG_PATTERN   = re.compile(r"#[A-Za-z0-9_]+", re.IGNORECASE)
NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z\s]", re.UNICODE)
MULTISPACE_PATTERN = re.compile(r"\s+")

def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = text.replace("\n", " ")
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    # hapus simbol # tapi pertahankan kata
    text = HASHTAG_PATTERN.sub(lambda m: m.group(0)[1:], text)
    text = text.lower() # casefolding
    text = NON_ALPHA_PATTERN.sub(" ", text) # cleaning karakter non huruf
    text = MULTISPACE_PATTERN.sub(" ", text)
    return text.strip()


def tokenize(text: str):
    return [t for t in text.split() if t]




def normalize_tokens(tokens, extra_norm: dict | None = None):
    norm = BASE_NORMALIZER.copy()
    if extra_norm:
        # kunci kamus dianggap sudah lower/stem bila perlu
        norm.update({str(k).lower(): str(v).lower() for k, v in extra_norm.items()})
    return [norm.get(tok, tok) for tok in tokens]



def remove_stopwords(tokens):
    return [t for t in tokens if t not in _STOPWORDS]




def stem_tokens(tokens):
    # Sastrawi stemmer bekerja pada kalimat; untuk presisi, stem per token
    return [_stemmer.stem(t) for t in tokens]




def run_preprocessing(df: pd.DataFrame, text_col: str, date_col: str | None = None,
                    extra_norm: dict | None = None) -> pd.DataFrame:
    out = df.copy()
    out['__raw_text__'] = out[text_col].astype(str)
    out['clean_text'] = out['__raw_text__'].apply(basic_clean)
    out['tokens'] = out['clean_text'].apply(tokenize)
    out['normalized'] = out['tokens'].apply(lambda toks: normalize_tokens(toks, extra_norm))
    out['no_stop'] = out['normalized'].apply(remove_stopwords)
    out['stemmed_tokens'] = out['no_stop'].apply(stem_tokens)
    out['stemmed_text'] = out['stemmed_tokens'].apply(lambda toks: " ".join(toks))


    if date_col and date_col in out.columns:
        out['__date__'] = out[date_col].apply(parse_date_safe)
    else:
        out['__date__'] = pd.NaT


    # kolom tampilan tahap‑demi‑tahap
    ordered_cols = [text_col]
    if date_col and date_col in df.columns: ordered_cols.append(date_col)
    ordered_cols += [
        '__raw_text__', 'clean_text', 'tokens', 'normalized', 'no_stop',
        'stemmed_tokens', 'stemmed_text', '__date__'
    ]
    return out[ordered_cols]