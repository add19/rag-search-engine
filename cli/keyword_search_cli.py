#!/usr/bin/env python3

import argparse
import json
from nltk.stem import PorterStemmer
from utils import preprocess_text
from utils import tokenize_input
from utils import load_stop_words

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