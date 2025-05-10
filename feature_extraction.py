import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from difflib import get_close_matches
import re
from collections import Counter

def load_data(filename="books_clean.csv"):
    """Enhanced data loading with improved genre extraction"""
    try:
        df = pd.read_csv(filename)

        for col in ['title', 'author', 'description']:
            if col not in df.columns:
                df[col] = f"Unknown {col.capitalize()}"

        df['title'] = df['title'].fillna("Unknown Title").str.strip()
        df['author'] = df['author'].fillna("Unknown Author").str.strip()
        df['description'] = df['description'].fillna("").str.strip()

        print("\nExtracting potential genres...")
        df['inferred_genres'] = df.apply(
            lambda x: infer_genres(x['title'], x['author'], x['description']),
            axis=1
        )

        print("\n=== Data Quality Report ===")
        print(f"Total books: {len(df)}")
        print(f"Unique titles: {df['title'].nunique()}")
        print(f"Unique authors: {df['author'].nunique()}")

        genre_counts = Counter([g for genres in df['inferred_genres'] for g in genres])
        print("\nTop 10 inferred genres:")
        for genre, count in genre_counts.most_common(10):
            print(f"{genre}: {count}")

        return df

    except Exception as e:
        print(f"\nERROR loading data: {e}")
        return None

def infer_genres(title, author, description):
    """Enhanced genre inference using multiple fields and more precise keywords"""
    
    genre_keywords = {
        'science fiction': ['time travel', 'space exploration', 'cyberpunk', 'alien invasion', 'robot', 'future', 'dystopia', 'galaxy', 'mars', 'virtual reality'],
        'fantasy': ['dragon', 'wizard', 'magical realm', 'sorcery', 'elf', 'kingdom', 'epic quest', 'mythical', 'spellcasting', 'troll'],
        'technology': ['machine learning', 'neural network', 'artificial intelligence', 'data science', 'robotics', 'blockchain', 'quantum computing', 'algorithm'],
        'classic': ['19th century', 'historical fiction', 'literary', 'classic literature', 'social commentary', 'Victorian', 'Hemingway', 'Dostoevsky', 'Dickens'],
        'physics': ['quantum mechanics', 'relativity', 'particle physics', 'cosmology', 'einstein', 'black hole', 'universe', 'big bang', 'gravity'],
        'mathematics': ['calculus', 'algebra', 'geometry', 'statistics', 'probability', 'mathematical theory', 'proof', 'equation'],
        'literature': ['novel', 'fiction', 'poetry', 'literary fiction', 'classic', 'drama', 'short story'],
        'adventure': ['expedition', 'journey', 'exploration', 'treasure hunt', 'quest', 'discovery', 'island', 'safari'],
        'historical fiction': ['ancient', 'medieval', 'Renaissance', 'war history', 'historical events', 'battle', 'kingdom', 'empire'],
        'horror': ['ghost', 'vampire', 'zombie', 'haunted', 'mystery', 'fear', 'suspense', 'supernatural'],
        'romance': ['love story', 'heartbreak', 'relationship', 'romantic drama', 'romantic comedy', 'passion', 'affair'],
        'thriller': ['mystery', 'suspense', 'detective', 'crime', 'psychological', 'twist', 'danger'],
        'biography': ['memoir', 'true story', 'life story', 'autobiography', 'historical figure', 'life of'],
        'children': ['kids', 'storybook', 'children', 'fairy tale', 'adventure', 'picture book'],
        'non-fiction': ['real story', 'documentary', 'informative', 'research', 'fact-based', 'educational', 'self-help'],
    }

    text = f"{title.lower()} {author.lower()} {description.lower()}"
    genres = []

    for genre, keywords in genre_keywords.items():
        if any(re.search(r'\b' + re.escape(keyword) + r'\b', text) for keyword in keywords):
            genres.append(genre)

    return genres if genres else ['General']

def enhance_features(df):
    """Create robust features with genre and author emphasis"""
    df = df.copy()

    df['boosted_text'] = df.apply(
        lambda x: f"{' '.join('genre_' + g.replace(' ', '_') for g in x['inferred_genres'])} "
                  f"author_{x['author'].lower().replace(' ', '_')} "
                  f"title_{x['title'].lower().replace(' ', '_')} "
                  f"{x['description']}",
        axis=1
    )

    df['boosted_text'] = (
        df['boosted_text'].str.lower()
        .str.replace(r'[^\w\s]', ' ', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )

    return df

def vectorize_features(df):
    """Feature engineering with optimized parameters"""
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            max_features=8000,
            stop_words='english',
            analyzer='word'
        )
        tfidf_matrix = vectorizer.fit_transform(df['boosted_text'])
        return tfidf_matrix, vectorizer

    except Exception as e:
        print(f"\nERROR in vectorization: {e}")
        return None, None

def recommend_books(title, df, similarity_matrix, top_n=5):
    """Enhanced recommendations with multiple strategies"""
    try:
        exact_matches = df[df['title'].str.lower() == title.lower()]
        if len(exact_matches) > 0:
            idx = exact_matches.index[0]
        else:
            partial_matches = df[df['title'].str.lower().str.contains(title.lower())]
            if len(partial_matches) > 0:
                idx = partial_matches.index[0]
            else:
                matches = get_close_matches(
                    title.lower(),
                    df['title'].str.lower(),
                    n=1,
                    cutoff=0.5
                )
                if not matches:
                    print(f"No matches found for '{title}'")
                    return None
                idx = df[df['title'].str.lower() == matches[0]].index[0]

        print(f"Found match: '{df.iloc[idx]['title']}'")

        sim_scores = list(enumerate(similarity_matrix[idx]))

        recommendations = []
        seen_titles = {df.iloc[idx]['title']}
        input_genres = set(df.iloc[idx]['inferred_genres'])
        input_author = df.iloc[idx]['author']

        for strategy in [
            ('same author + genre', lambda c: c['author'] == input_author and set(c['inferred_genres']).intersection(input_genres)),
            ('same genre', lambda c: set(c['inferred_genres']).intersection(input_genres)),
            ('same author', lambda c: c['author'] == input_author),
            ('general similarity', lambda c: True)
        ]:
            label, condition = strategy
            for i, score in sorted(sim_scores, key=lambda x: x[1], reverse=True):
                current = df.iloc[i]
                if current['title'] not in seen_titles and condition(current):
                    recommendations.append({
                        'title': current['title'],
                        'author': current['author'],
                        'genres': ', '.join(current['inferred_genres']),
                        'similarity_score': f"{score:.2f}",
                        'match_type': label
                    })
                    seen_titles.add(current['title'])
                    if len(recommendations) >= top_n:
                        break
            if len(recommendations) >= top_n:
                break

        return pd.DataFrame(recommendations).head(top_n)

    except Exception as e:
        print(f"\nERROR in recommendation: {e}")
        return None

if __name__ == "__main__":
    print("=== Advanced Book Recommendation System ===")
    print("Loading and processing data...")
    df = load_data()
    if df is None:
        exit("\nFailed to load data, exiting...")

    print("\nCreating features...")
    df = enhance_features(df)

    print("\nVectorizing text...")
    feature_matrix, vectorizer = vectorize_features(df)
    if feature_matrix is None:
        exit("\nFailed to create features, exiting...")

    print("\nCalculating similarities...")
    sim_matrix = cosine_similarity(feature_matrix)

    test_book = None
    for title in ["The Time Machine", "The War of the Worlds", "Dune"]:
        if title in df['title'].values:
            test_book = title
            break
    if test_book is None:
        test_book = df.iloc[0]['title']

    print(f"\nTesting recommendations for: '{test_book}'")

    recs = recommend_books(test_book, df, sim_matrix)

    if recs is not None:
        print("\n=== Recommendations ===")
        print(recs[['title', 'author', 'genres', 'similarity_score', 'match_type']].to_string(index=False))
    else:
        print("\nNo recommendations generated - showing random sample")
        print(df.sample(min(5, len(df)))[['title', 'author', 'inferred_genres']].to_string(index=False))
