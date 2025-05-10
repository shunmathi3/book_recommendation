import requests
import json

def fetch_books(query="machine learning", limit=50):
    url = "http://openlibrary.org/search.json"
    params = {"q": query, "limit": limit}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get("docs", [])
    except Exception as e:
        print(f"Error fetching books: {e}")
        return []

def save_raw_data(books, filename="books_raw.json"):
    try:
        with open(filename, "w") as f:
            json.dump(books, f)
        print(f"Successfully saved {len(books)} books to {filename}")
    except Exception as e:
        print(f"Error saving data: {e}")

if __name__ == "__main__":
    print("Fetching books from Open Library...")
    books = fetch_books()
    if books:
        print(f"First book title: {books[0].get('title', 'No title')}")
        save_raw_data(books)
    else:
        print("No books were fetched. Check your internet connection.")