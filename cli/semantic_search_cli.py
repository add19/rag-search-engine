#!/usr/bin/env python3

import argparse
from lib.semantic_search import SemanticSearch, verify_model, embed_text, verify_embeddings, embed_query_text
from utils import load_movies

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verifies the model has built")

    embed_parser = subparsers.add_parser("embed_text", help="Generates embedding for a given word")
    embed_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("verify_embeddings", help="Verifies and builds/loads embeddings")

    embed_query_parser = subparsers.add_parser("embedquery", help="Generates embedding for a given word")
    embed_query_parser.add_argument("query", type=str, help="Search query")

    search_query_parser = subparsers.add_parser("search", help="Search based on cosine similarity")
    search_query_parser.add_argument("query", type=str, help="Search query")
    search_query_parser.add_argument("--limit", type=int, default=5, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search = SemanticSearch()
            movies = load_movies()
            search.load_or_create_embeddings(movies["movies"])
            top_res = search.search(args.query, args.limit)
            idx = 1
            for res in top_res:
                print(f"{idx}. {res["title"]} (score: {res["score"]})\n {res["description"]}")
                idx += 1
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()