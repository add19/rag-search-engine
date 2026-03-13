import json
import pickle

from utils import preprocess_text
from utils import tokenize_input
from utils import load_stop_words
from nltk.stem import PorterStemmer
from pathlib import Path

class InvertedIndex:
    index = {} # mapping of tokens to sets of document ids
    docmap = {} # document ids to their full document objects
    stemmer = None

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.index = {}
        self.docmap = {}

    def __add_document(self, doc_id, text):
        # tokenize the input text and return them as list
        preprocessed_text = preprocess_text(text)
        stop_words = load_stop_words()

        text_tokens = tokenize_input(preprocessed_text, stop_words, self.stemmer)
        for token in text_tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

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

        return
    
    def load(self):
        index_path = Path("cache/index.pkl")
        docmap_path = Path("cache/docmap.pkl")

        if not index_path.exists():
            raise FileNotFoundError(f"Path does not exist: {index_path}")
        if not docmap_path.exists():
            raise FileNotFoundError(f"Path does not exist: {docmap_path}")
        
        with open('cache/index.pkl', 'rb') as f:
            self.index = pickle.load(f)

        with open('cache/docmap.pkl', 'rb') as f:
            self.docmap = pickle.load(f)