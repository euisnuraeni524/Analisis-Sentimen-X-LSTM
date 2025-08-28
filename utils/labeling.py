import pandas as pd


POL_MAP = { -1: 'Negative', 0: 'Neutral', 1: 'Positive' }
REV_POL_MAP = {v:k for k, v in POL_MAP.items()}




def load_lexicon_csv(file) -> dict:
    """Terima CSV dengan kolom: word, polarity (−1/0/1). Return dict: {word: polarity}.
    Disarankan kata dalam bentuk stem. Akan di-lowercase saat load.
    """
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    if 'word' not in df.columns or 'polarity' not in df.columns:
        raise ValueError("Lexicon CSV harus memiliki kolom 'word' dan 'polarity'.")
    lex = {}
    for _, row in df.iterrows():
        w = str(row['word']).strip().lower()
        try:
            p = int(row['polarity'])
        except Exception:
            p = 0
        p = -1 if p < 0 else (1 if p > 0 else 0)
        lex[w] = p
    return lex




def score_tokens(tokens, lexicon: dict) -> int:
    score = 0
    for t in tokens:
        score += lexicon.get(t, 0)
    # threshold sederhana: >0 pos, <0 neg, ==0 netral
    return 1 if score > 0 else (-1 if score < 0 else 0)




def label_dataframe(df_proc: pd.DataFrame, token_col: str, lexicon: dict) -> pd.DataFrame:
    out = df_proc.copy()
    out['lexicon_score'] = out[token_col].apply(lambda toks: score_tokens(toks, lexicon))
    out['label'] = out['lexicon_score'].map(POL_MAP)
    return out

