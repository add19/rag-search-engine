import json
import math
import string
from pathlib import Path
from nltk.stem import PorterStemmer
from collections import Counter
import json
import pickle

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def tokenize_input(text):
    stemmer = PorterStemmer()
    stop_words = load_stop_words()

    text_tokens = text.split()
    text_tokens = [stemmer.stem(t) for t in text_tokens if t and t not in stop_words]
    return text_tokens

def calculate_tf(doc_id, term):
    index = InvertedIndex()
    index.load()
    return index.get_tf(doc_id, term)

def calculate_idf(term):
    index = InvertedIndex()
    index.load()
    total_doc_count = len(index.docmap)
    preprocessed_query = preprocess_text(term)
    query_tokens = tokenize_input(preprocessed_query)
    matching_docs = index.get_documents(query_tokens[0])
    term_match_doc_count = len(matching_docs)
    inverse_doc_freq = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
    return inverse_doc_freq

def calculate_tf_idf(doc_id, query):
    preprocessed_query = preprocess_text(query)
    query_tokens = tokenize_input(preprocessed_query)
    tf_idf_score = 0

    for token in query_tokens:
        tf = calculate_tf(doc_id, token)
        idf = calculate_idf(token)
        tf_idf_score += tf * idf
    
    return tf_idf_score

def search(query):
    results = []
    try:
        index = InvertedIndex()
        index.load()
    except Exception:
        raise Exception("Cannot load index")
                
    preprocessed_query = preprocess_text(query)
    query_tokens = tokenize_input(preprocessed_query)

    for token in query_tokens:
        docs = index.get_documents(token)
        for doc in docs:
            results.append(doc)
            if len(results) == 5:
                break
        if len(results) == 5:
            break
    
    return results

def load_stop_words():
    with open('data/stopwords.txt', 'r') as f:
        data = f.read()
        stop_words = data.splitlines()
    return stop_words

def load_movies():
    with open('data/movies.json', 'r') as f:
        movies_obj = json.load(f)
    
    return movies_obj

class InvertedIndex:
    index = {} # mapping of tokens to sets of document ids
    docmap = {} # document ids to their full document objects
    stemmer = None
    term_frequency = {}

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.index = {}
        self.docmap = {}
        self.term_frequency = {}

    def __add_document(self, doc_id, text):
        # tokenize the input text and return them as list
        preprocessed_text = preprocess_text(text)
        stop_words = load_stop_words()

        text_tokens = tokenize_input(preprocessed_text, stop_words, self.stemmer)
        for token in text_tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

            if doc_id not in self.term_frequency:
                self.term_frequency[doc_id] = Counter()

            self.term_frequency[doc_id].update([token])
        return
    
    def get_documents(self, term):
        # get the set of document ids for the given token and return them as a sorted list
        if term.lower() not in self.index:
            return []
        
        doc_id_set = self.index[term.lower()]
        ret_list = []
        for doc_id in doc_id_set:
            ret_list.append(doc_id)
        sorted_ret_list = sorted(ret_list)
        return sorted_ret_list
    
    def get_tf(self, doc_id, term):
        if doc_id not in self.term_frequency:
            return 0
        return self.term_frequency[doc_id][term]

    def build(self):
        # iterate over all the movies and add them to both the index and the docmap
        with open('data/movies.json', 'r') as f:
            movies_obj = json.load(f)
        
        for movie_obj in movies_obj["movies"]:
            self.__add_document(movie_obj["id"], f"{movie_obj['title']} {movie_obj['description']}")
            self.docmap[movie_obj["id"]] = movie_obj
        return

    def save(self):
        # save the index and the docmap attributes to disk using the pickle module's dump method
        path = Path("cache")
        path.mkdir(exist_ok=True)

        pickle.dump(self.index, open('cache/index.pkl', 'wb'))
        pickle.dump(self.docmap, open('cache/docmap.pkl', 'wb'))
        pickle.dump(self.term_frequency, open('cache/term_frequencies.pkl', 'wb'))

        return
    
    def load(self):
        index_path = Path("cache/index.pkl")
        docmap_path = Path("cache/docmap.pkl")
        tf_path = Path("cache/term_frequencies.pkl")

        if not index_path.exists():
            raise FileNotFoundError(f"Path does not exist: {index_path}")
        if not docmap_path.exists():
            raise FileNotFoundError(f"Path does not exist: {docmap_path}")
        if not tf_path.exists():
            raise FileNotFoundError(f"Path does not exist: {tf_path}")
        
        with open('cache/index.pkl', 'rb') as f:
            self.index = pickle.load(f)

        with open('cache/docmap.pkl', 'rb') as f:
            self.docmap = pickle.load(f)

        with open('cache/term_frequencies.pkl', 'rb') as f:
            self.term_frequency = pickle.load(f)