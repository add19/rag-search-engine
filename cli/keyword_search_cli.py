#!/usr/bin/env python3

import argparse
from pathlib import Path

from config import BM25_B, BM25_K1
from utils import calculate_tf
from utils import calculate_idf
from utils import search
from utils import calculate_tf_idf
from utils import get_bm25_idf
from utils import get_bm25_tf
from utils import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("build", help="Build inverted index")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    
    bm25_tf_parser = subparsers.add_parser(
    "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")
    
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    tf_idf_parser = subparsers.add_parser("tfidf", help="Get tf-idf score of a given search query")
    tf_idf_parser.add_argument("doc_id", type=int, help="document id")
    tf_idf_parser.add_argument("query", type=str, help="search query")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency of a given term")
    idf_parser.add_argument("term", type=str, help="search term")
    
    tf_parser = subparsers.add_parser("tf", help="Get term frequency of a given term")
    tf_parser.add_argument("doc_id", type=int, help="document id")
    tf_parser.add_argument("term", type=str, help="Search term")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    
    
    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            
            results = search(args.query)
            
            for i in range(0, len(results), 1):
                print(f"{index.docmap[results[i]]["id"]} {index.docmap[results[i]]["title"]}")
            pass
        case "build":
            index = InvertedIndex()
            index.build()
            index.save()
            pass
        case "tf":
            print(f"doc_id: {args.doc_id} tf: {calculate_tf(args.doc_id, args.term)}")
            pass
        case "idf":
            inverse_doc_freq = calculate_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {inverse_doc_freq:.2f}")
            pass
        case "tfidf":
            tf_idf_score = calculate_tf_idf(args.doc_id, args.query)
            print(f"TF-IDF score of '{args.query}' in document '{args.doc_id}': {tf_idf_score:.2f}")
        case "bm25idf":
            bm25idf = get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = get_bm25_tf(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25search":
            index = InvertedIndex()
            index.load()
            docs = index.bm25_search(args.query, 5)
            idx = 1
            for doc_id, score in docs.items():
                print(f"{idx}. ({doc_id}) {index.docmap[doc_id]["title"]} - Score: {score:.2f}")
                idx += 1
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
