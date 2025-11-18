import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import numpy as np
import os
import json
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess(text):
    tokens = [t for t in simple_preprocess(text) if t not in STOPWORDS]
    return tokens

def vectorize(tokens, model, vector_size):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

def create_vectorized_dataset(news_df, impact_csv='datasets/historical_prices_impact.csv', model=None, out_csv='datasets/vectorized_news_skip_gram_embeddings.csv'):
    """Merge `news_df` (must contain `date`, `symbol` and `news`/text column) with `impact_csv` and save dataset.

    The output CSV will have columns: date, symbol, news_vector (JSON), impact_score.
    """
    if not os.path.exists(impact_csv):
        raise FileNotFoundError(f"Impact CSV not found: {impact_csv}")

    df_news = news_df.copy()
    df_impact = pd.read_csv(impact_csv)

    # normalize names
    df_news.columns = [c.strip() for c in df_news.columns]
    df_impact.columns = [c.strip() for c in df_impact.columns]

    # detect text column if necessary
    if 'news' in df_news.columns:
        text_col = 'news'
    else:
        object_cols = [c for c in df_news.columns if df_news[c].dtype == object]
        object_cols = [c for c in object_cols if c not in ('symbol', 'date')]
        if not object_cols:
            raise ValueError('No text-like column found in news DataFrame; please provide one named `news` or another string column')
        text_col = object_cols[0]

    # Ensure tokens and vectors exist; if not, compute
    if 'tokens' not in df_news.columns:
        df_news['tokens'] = df_news[text_col].fillna('').astype(str).apply(preprocess)
    if 'news_vector' not in df_news.columns:
        vector_size_local = model.vector_size
        df_news['news_vector'] = df_news['tokens'].apply(lambda toks: np.zeros(vector_size_local) if len(toks)==0 else np.mean([model.wv[w] for w in toks if w in model.wv], axis=0))

    # parse dates to date-only for merge
    df_news['date'] = pd.to_datetime(df_news['date'], errors='coerce')
    df_impact['date'] = pd.to_datetime(df_impact['date'], errors='coerce')
    df_news['date_only'] = df_news['date'].dt.date
    df_impact['date_only'] = df_impact['date'].dt.date

    merged = pd.merge(
        df_news,
        df_impact[['date_only', 'symbol', 'impact_score']],
        left_on=['date_only', 'symbol'],
        right_on=['date_only', 'symbol'],
        how='left',
    )
    # Remove rows without impact_score (these are news rows that did not match any impact record)
    missing = merged['impact_score'].isna().sum()
    if missing > 0:
        print(f'Warning: {missing} rows have no matching impact_score and will be dropped from the output CSV')
    merged = merged.dropna(subset=['impact_score'])

    out_df = pd.DataFrame()
    out_df['date'] = merged['date'].dt.strftime('%Y-%m-%d')
    out_df['symbol'] = merged['symbol']
    out_df['news_vector'] = merged['news_vector'].apply(lambda v: json.dumps(v.tolist()))
    out_df['impact_score'] = merged['impact_score']

    out_df.to_csv(out_csv, index=False)
    print(f'Saved {out_csv} ({len(out_df)} rows)')



def skip_gram_embeddings():
    # --- Load data ---
    df = pd.read_csv('datasets/aggregated_news.csv')


    # determine text column (adjust if your column has another name)
    if 'news' in df.columns:
        text_col = 'news'
    else:
        object_cols = [c for c in df.columns if df[c].dtype == object]
        if not object_cols:
            raise ValueError('No text-like column found in CSV. Please set `text_col` manually.')
        text_col = object_cols[0]

    vector_size = 100
    window = 5
    min_count = 2
    workers = 4
    epochs = 5
    seed = 42

    df['tokens'] = df[text_col].fillna('').astype(str).apply(preprocess)

    tokenized_docs = [tokens for tokens in df['tokens'] if len(tokens) > 0]

    # Train Skip-gram Word2Vec (sg=1)
    w2v_model = gensim.models.Word2Vec(
        sentences=tokenized_docs,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,  # use Skip-gram as requested
        seed=seed,
    )
    # ensure training runs for the chosen number of epochs
    w2v_model.train(tokenized_docs, total_examples=len(tokenized_docs), epochs=epochs)

    # Save model and vectors
    w2v_model.save('word2vec.model')
    w2v_model.wv.save_word2vec_format('word2vec.kv', binary=False)

    # --- Build TF-IDF weighted document vectors ---
    print('Building TF-IDF vectorizer on raw texts...')
    texts = df[text_col].fillna('').astype(str).tolist()
    tfidf = TfidfVectorizer(min_df=2)
    tfidf.fit(texts)
    idf_dict = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))

    def tfidf_weighted_vector(tokens, model, vector_size, idf_dict):
        vecs = []
        weights = []
        for t in tokens:
            if t in model.wv:
                w = idf_dict.get(t, 1.0)
                vecs.append(model.wv[t] * w)
                weights.append(w)
        if len(vecs) == 0:
            return np.zeros(vector_size)
        return np.sum(vecs, axis=0) / (np.sum(weights) + 1e-9)

    X = np.array([tfidf_weighted_vector(tokens, w2v_model, vector_size, idf_dict) for tokens in df['tokens']])
    print('Feature matrix shape:', X.shape)
    np.save('doc_vectors.npy', X)
    print("Saved document vectors to 'doc_vectors.npy'")

    # Reuse computed vectors: attach to DataFrame so create_vectorized_dataset won't recompute
    df['news_vector'] = list(X)

    # create dataset using the df we already loaded and the trained model
    create_vectorized_dataset(df, impact_csv='datasets/historical_prices_impact.csv', model=w2v_model, out_csv='datasets/vectorized_news_skip_gram_embeddings.csv')

if __name__ == '__main__':
    skip_gram_embeddings()

