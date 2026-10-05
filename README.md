# IMDB / Netflix Shows Content-Based Recommender

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit-learn-NLP-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

This repository builds a content-based recommendation system using the IMDB show dataset based on genres, directors, and cast features[cite: 18].

---

## Project Workflow
1. **Data Preprocessing**: Loading processed dataset (`imdb_processed.csv`)[cite: 18] and handling missing values across genre, director, and cast columns[cite: 18].
2. **Metadata Merging (Soup Creation)**: Combining multiple categorical features and titles into a single textual representation (`soup`)[cite: 18].
3. **Text Vectorization**: Extracting features from the soup column using `TfidfVectorizer` with stop words removal[cite: 18].
4. **Similarity Computation**: Calculating similarity matrices using `linear_kernel` (cosine similarity)[cite: 18].
5. **Recommendation Engine**: Mapping show titles to indices and fetching top similar recommendations[cite: 18].
6. **Web Application**: Interactive deployment interface built with Streamlit.

---

## Getting Started & Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/imdb-shows-recommender.git](https://github.com/YOUR_USERNAME/imdb-shows-recommender.git)
   cd imdb-shows-recommender
