# Book Recommendation System
This project is not completed 
# Overview
This project is a book recommendation system that fetches book data from Open Library, processes and cleans the data, and provides personalized book recommendations based on content similarity, genres, and authors.

# Features
Data Fetching: Retrieves book data from Open Library API

Data Cleaning: Processes raw book data into a structured format

Genre Inference: Automatically categorizes books into genres based on content analysis

Content-Based Recommendations: Recommends books based on:

# Text similarity (using TF-IDF and cosine similarity)

Genre matching

Author similarity

Multi-Stage Matching: Uses exact, partial, and fuzzy matching for robust title recognition

# Usage
First, fetch book data:
    python fetch_books.py

Clean and preprocess the data:
  python clean_books.py
  
Get book recommendations:
  python feature_extraction.py
  
The system will automatically test with sample books like "The Time Machine" or "Dune"

# Customization
You can modify the system by:

Changing the default search query in fetch_books.py

Adjusting genre keywords in feature_extraction.py

Modifying the recommendation strategies in recommend_books()

# Example Output
=== Advanced Book Recommendation System ===
Loading and processing data...

Extracting potential genres...

=== Data Quality Report ===
Total books: 50
Unique titles: 50
Unique authors: 42

Top 10 inferred genres:
science fiction: 18
fantasy: 12
technology: 8
physics: 6
adventure: 5
...

Testing recommendations for: 'The Time Machine'

=== Recommendations ===
Title                     Author            Genres                     Similarity Score  Match Type
The War of the Worlds     H.G. Wells       science fiction, adventure  0.87              same author + genre
