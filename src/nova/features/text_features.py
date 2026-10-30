from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf():
    # Fit vectorizer trên tập train chỉ; không fit trên toàn bộ dữ liệu trước khi chia tập.
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
    )
