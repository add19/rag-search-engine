#!/usr/bin/env python3

import argparse
import json
import string
from nltk.stem import PorterStemmer

def preprocess_text(text):
    text = text.lower()
    # translate_table = str.maketrans("", "", ",!")
    # text = text.translate(translate_table)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def tokenize_input(text, stop_words, stemmer):
    text_tokens = text.split()
    text_tokens = [stemmer.stem(t) for t in text_tokens if t and t not in stop_words]
    return text_tokens

def load_stop_words():
    with open('data/stopwords.txt', 'r') as f:
        data = f.read()
        stop_words = data.splitlines()
    return stop_words

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    
    stemmer = PorterStemmer()
    
    args = parser.parse_args()
    stop_words = load_stop_words()

    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            
            with open('data/movies.json', 'r') as f:
                movies_obj = json.load(f)
            
            results = []
            
            preprocessed_query = preprocess_text(args.query)
            query_tokens = tokenize_input(preprocessed_query, stop_words, stemmer)

            for movie_obj in movies_obj["movies"]:
                preprocessed_movie_title = preprocess_text(movie_obj["title"])
                title_tokens = tokenize_input(preprocessed_movie_title, stop_words, stemmer)

                for query_token in query_tokens:
                    for title_token in title_tokens:
                        if query_token in title_token:
                            results.append(movie_obj["title"])

            for i in range(0, len(results), 1):
                print(f"{i + 1}. {results[i]}")

            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()