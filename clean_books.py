import pandas as pd
import spacy
import json

def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except:
        print("Spacy model not found. Installing...")
        import os
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_spacy()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

def clean_data(raw_file="books_raw.json", output_file="books_clean.csv"):
    try:
        with open(raw_file, "r") as f:
            books = json.load(f)
        
        df = pd.DataFrame(books)

        df['title'] = df['title'].fillna("").str.strip()
        df['author'] = df.get('author_name', [[]]).apply(lambda x: x[0] if isinstance(x, list) and x else "Unknown Author")
        df['description'] = df.get('first_sentence', "").apply(lambda x: x.get('value', "") if isinstance(x, dict) else x)

        df['description'] = df['description'].apply(preprocess_text)
        df = df[['title', 'author', 'description']]
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")

    except Exception as e:
        print(f"Error cleaning data: {e}")


    cleaned = []
    for book in books:
        first_sentence = book.get("first_sentence", ["No description"])[0]
        cleaned.append({
            "title": book.get("title", "Unknown"),
            "author": ", ".join(book.get("author_name", ["Unknown"])),
            "description": preprocess_text(first_sentence),
            "genres": book.get("subject", [])[:3],
            "year": book.get("first_publish_year", "Unknown"),
        })

    df = pd.DataFrame(cleaned)
    df.to_csv("books_clean.csv", index=False)
    print(f"Successfully cleaned {len(df)} books. Saved to books_clean.csv")
    return df

if __name__ == "__main__":
    print("Cleaning book data...")
    df = clean_data()
    if df is not None:
        print("\nSample cleaned book:")
        print(df.iloc[0].to_dict())