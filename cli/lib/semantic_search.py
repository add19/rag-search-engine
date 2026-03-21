import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
import numpy as np

def verify_model():
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")

def embed_text(text):
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    search = SemanticSearch()
    with open('data/movies.json', 'r') as f:
        movies_obj = json.load(f)

    embeddings = search.load_or_create_embeddings(movies_obj["movies"])
    print(f"Number of docs:   {len(search.documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if len(text) == 0 or text == " ":
            raise ValueError("Cannot generate embedding for empty string")
        
        result = self.model.encode([text])
        return result[0]
    
    def build_embeddings(self, documents):
        self.documents = documents
        doc_list = []
        
        for document in documents:
            self.document_map[document["id"]] = document
            doc_list.append(f"{document['title']}: {document['description']}")

        self.embeddings = self.model.encode(doc_list, show_progress_bar = True)
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings
    
    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)
        tuples = []
        idx = 0
        for doc in self.documents:
            cos_sim = cosine_similarity(query_embedding, self.embeddings[idx])
            idx += 1
            tuples.append((cos_sim, doc))

        sorted_res = sorted(tuples, key=lambda x:x[0], reverse=True)
        result = []
        for i in range(limit):
            dict_item = {}
            dict_item["score"] = sorted_res[i][0]
            dict_item["title"] = sorted_res[i][1]["title"]
            dict_item["description"] = sorted_res[i][1]["description"]
            result.append(dict_item)
        return result

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document

        embeddings_path = Path("cache/movie_embeddings.npy")

        if not embeddings_path.exists():
            return self.build_embeddings(documents)

        self.embeddings = np.load(embeddings_path)
        if len(documents) == len(self.embeddings):
            return self.embeddings
        
        return self.build_embeddings(documents)