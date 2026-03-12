import string

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
