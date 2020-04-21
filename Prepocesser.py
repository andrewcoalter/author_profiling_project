import os
from html.parser import HTMLParser
import re
import numpy as np
import torch

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r', encoding="utf-8")
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model


class TrainingDataHTMLParser(HTMLParser):
    data = []

    def unknown_decl(self, data):
        self.data.append(data[6:].lower().replace('\t', '').replace('\n', ''))


def fetch_author_truths(test):
    if test:
        file_name = "test_truth.txt"
    else:
        file_name = "train_truth.txt"
    truths = open(file_name, "r")
    author_gender = {}
    author_age = {}
    for line in truths:
        information = line.split(":::")
        author_gender[information[0]] = information[1]
        author_age[information[0]] = information[2]
    ordered_authors = sorted(author_age.keys())
    age_tensors = []
    gender_tensors = []
    for author in ordered_authors:
        age_category = int(author_age[author][:2])
        age_tensors.append(int(age_category == 25) + 2 * int(age_category == 35) + 3 * int(age_category == 50))
        gender_tensors.append(int(author_gender[author] == "M"))
    return [torch.tensor(age_tensors), torch.tensor(gender_tensors)]


parser = TrainingDataHTMLParser()


def get_tokens():
    token_counts = {}
    longest_token_set = 0
    file_locations = ["test", "train"]
    for file_location in file_locations:
        for file in os.scandir(file_location):
            file = open(file.path, encoding="utf8")
            text = file.read()
            parser.data = []
            parser.feed(text)
            texts = parser.data
            token_set = 0
            for text in texts:
                token_set += 1
                tokens = tokenize(text)
                for token in tokens:
                    token_set += 1
                    if token in token_counts:
                        token_counts[token] += 1
                    else:
                        token_counts[token] = 1
            token_set -= 1
            if token_set > longest_token_set:
                longest_token_set = token_set
    token_numbering = {}
    i = 0
    for token in token_counts:
        if token_counts[token] >= 3:
            token_numbering[token] = i
            i += 1
    return token_numbering, longest_token_set


def fetch_author_tweets(test):
    author_tweets = {}
    if test:
        file_name = "test"
    else:
        file_name = "train"
    for file in os.scandir(file_name):
        author = file.name.replace(".xml", "")
        file = open(file.path, encoding="utf8")
        author_tweets[author] = file.read()
        parser.data = []
        parser.feed(author_tweets[author])
        author_tweets[author] = parser.data
    return author_tweets

FLAGS = re.MULTILINE | re.DOTALL

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = " <hashtag> {} <allcaps> ".format(hashtag_body)
    else:
        result = " ".join([" <hashtag> "] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
    return result

def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps>"


def preprocess(text):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " <url> ")
    text = re_sub(r"/"," / ")
    text = re_sub(r"@\w+", " <user> ")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), " <smile> ")
    text = re_sub(r"{}{}p+".format(eyes, nose), " <lolface> ")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), " <sadface> ")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), " <neutralface> ")
    text = re_sub(r"<3"," <heart> ")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", " <number> ")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat> ")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong> ")
    text = re_sub(r"([A-Z]){2,}", allcaps)
    return text.lower()


def tokenize(text):
    text = preprocess(text)
    tokens = []
    current_token = ""
    previous_was_alnum = False
    for char in text:
        if not char.isalnum():
            if (current_token != "" and char != "<" and char != ">") and (previous_was_alnum or current_token[len(current_token) - 1] != char):
                tokens.append(current_token)
                current_token = ""
            previous_was_alnum = False
            if char == "<" or char == ">":
                previous_was_alnum = True
        else:
            if current_token != "" and not previous_was_alnum:
                tokens.append(current_token)
                current_token = ""
            previous_was_alnum = True
        if not char.isspace():
            current_token = current_token + char
    if current_token != "":
        tokens.append(current_token)
    return tokens


def fetch_author_tweets_tokens(test):
    author_tweets = fetch_author_tweets(test)
    author_tweets_tokens = {}
    for author in author_tweets:
        tweets = author_tweets[author]
        tweets_tokens = []
        for tweet in tweets:
            tweet_tokens = tokenize(tweet)
            tweets_tokens.append(tweet_tokens)
        author_tweets_tokens[author] = tweets_tokens
    token_numbering, _ = get_tokens()
    ordered_authors = sorted(author_tweets_tokens.keys())
    tensors = []
    for author in ordered_authors:
        tensor = np.zeros(len(token_numbering.keys()))
        for tweets in author_tweets_tokens[author]:
            for token in tweets:
                if token in token_numbering:
                    tensor[token_numbering[token]] += 1
        tensors.append(tensor)
    return torch.Tensor(tensors)


def fetch_number_of_authors(test, i):
    return len(fetch_author_truths(test)[i])


Word2VecModel = loadGloveModel("glove.twitter.27B.200d.txt")

AUTHOR_TWEETS = (fetch_author_tweets(False), fetch_author_tweets(True))
ORDERED_AUTHORS = []
AUTHOR_TWEETS_TOKENS = []
for author_tweets in AUTHOR_TWEETS:
    author_tweets_tokens = {}
    for author in author_tweets:
        tweets = author_tweets[author]
        tweets_tokens = []
        for tweet in tweets:
            tweet_tokens = tokenize(tweet)
            tweets_tokens.append(tweet_tokens)
        author_tweets_tokens[author] = tweets_tokens
    AUTHOR_TWEETS_TOKENS.append(author_tweets_tokens)
    ORDERED_AUTHORS.append(sorted(author_tweets_tokens.keys()))

token_numbering, longest_token_sequence = get_tokens()

TOKEN_TENSORS = []
for i, author_tweets in enumerate(AUTHOR_TWEETS_TOKENS):
    token_tensors = {}
    for author in ORDERED_AUTHORS[i]:
        for tweets in author_tweets[author]:
            for token in tweets:
                if token.lower() in Word2VecModel:
                    token_tensors[token] = Word2VecModel[token.lower()]
    token_tensors[" "] = np.zeros(200)
    TOKEN_TENSORS.append(token_tensors)

def fetch_word2vec_model():
    return Word2VecModel

def fetch_author_tweets_tokens_ordered(test):
    hit = 0
    miss = 0
    authors = ORDERED_AUTHORS[test]
    tensors = []
    for author in authors:
        tensor = []
        author_tweets_tokens = AUTHOR_TWEETS_TOKENS[test]
        token_tensors = TOKEN_TENSORS[test]
        for tweets in author_tweets_tokens[author]:
            for token in tweets:
                if token.lower() in token_tensors:
                    miss += 1
                    tensor.append(token_tensors[token.lower()])
            hit += 1
            tensor.append(token_tensors[" "])
        i = len(tensor)
        while i < longest_token_sequence:
            hit += 1
            tensor.append(token_tensors[" "])
            i += 1
        tensors.append(tensor)
    print(hit/(hit + miss))
    return torch.Tensor(tensors)


def fetch_author_tweets_tokens_ordered_singular(test, i):
    author = ORDERED_AUTHORS[test][i]
    tensors = []
    tensor = []
    author_tweets_tokens = AUTHOR_TWEETS_TOKENS[test]
    token_tensors = TOKEN_TENSORS[test]
    for tweets in author_tweets_tokens[author]:
        for token in tweets:
            if token in token_tensors:
                tensor.append(token_tensors[token])
        tensor.append(token_tensors[" "])
    i = len(tensor)
    while i < longest_token_sequence:
        tensor.append(token_tensors[" "])
        i += 1
    tensors.append(tensor)
    return torch.Tensor(tensors)
