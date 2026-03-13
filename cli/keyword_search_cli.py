#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

CLI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CLI_DIR.parent

for path in (CLI_DIR, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from nltk.stem import PorterStemmer
from utils import load_movies, preprocess_text
from utils import tokenize_input
from utils import load_stop_words
from Index.InvertedIndex import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("build", help="Build inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    
    stemmer = PorterStemmer()
    
    args = parser.parse_args()
    stop_words = load_stop_words()

    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            
            results = []
            try:
                index = InvertedIndex()
                index.load()
            except Exception:
                raise Exception("Cannot load index")
                        
            preprocessed_query = preprocess_text(args.query)
            query_tokens = tokenize_input(preprocessed_query, stop_words, stemmer)

            for token in query_tokens:
                docs = index.get_documents(token)
                for doc in docs:
                    results.append(doc)
                    if len(results) == 5:
                        break
                if len(results) == 5:
                    break
            
            for i in range(0, len(results), 1):
                print(f"{index.docmap[results[i]]["id"]} {index.docmap[results[i]]["title"]}")

            pass
        case "build":
            index = InvertedIndex()
            index.build()
            index.save()
            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
