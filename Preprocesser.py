import os
from html.parser import HTMLParser
import re
import shutil
import numpy as np
import torch


class TrainingDataHTMLParser(HTMLParser):
    data = []

    def unknown_decl(self, data):
        self.data.append(data[6:].lower().replace('\t', '').replace('\n', ''))


parser = TrainingDataHTMLParser()


# Load the Glove Model and convert it into a lookup table
def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile, 'r', encoding="utf-8")
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


# Reads the relevant truth file and returns the requested information in tensor form
def fetch_author_truths(test, validation):
    if test:
        if validation:
            file_name = "val/val_truth.txt"
        else:
            file_name = "test/test_truth.txt"
    else:
        file_name = "train/train_truth.txt"
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


# Reads the relevant truth file for independence testing and returns the requested information in tensor form
def fetch_author_truths_ind(social, review):
    if social:
        file_name = "ind/social/truth.txt"
    else:
        if review:
            file_name = "ind/review/truth.txt"
        else:
            file_name = "ind/blog/truth.txt"
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
        age_tensors.append(int(age_category == 25) + 2 * int(age_category == 35) + 3 * int(age_category == 50 or age_category == 65))
        gender_tensors.append(int(author_gender[author] == "MALE"))
    return [torch.tensor(age_tensors), torch.tensor(gender_tensors)]


# assuming a train/test split, this method changes balanced half of testing set into validation set
def generate_validation_set():
    test_file_name = "test/test_truth.txt"
    val_file_name = " val/val_truth.txt"
    truths = open(test_file_name, "r")
    lines = truths.readlines()
    validation = np.zeros([2, 4])
    move_line = np.zeros([len(lines)])
    for i, line in enumerate(lines):
        information = line.split(":::")
        author_gender_info = information[1]
        author_age_info = information[2]
        age_category = int(author_age_info[:2])
        author_age = int(age_category == 25) + 2 * int(age_category == 35) + 3 * int(age_category == 50)
        author_gender = int(author_gender_info == "M")
        move = validation[author_gender][author_age]
        if move:
            move_line[i] = 1
            validation[author_gender][author_age] = 0
        else:
            validation[author_gender][author_age] = 1
    test = open(test_file_name, "w+")
    val = open(val_file_name, "w+")
    for i, line in enumerate(lines):
        if move_line[i]:
            val.write(line)
            information = line.split(":::")
            author_name = information[0]
            shutil.move("test/" + author_name + ".xml", "val/" + author_name + ".xml")
        else:
            test.write(line)
    return True


# returns set of relevant tokens with numberings
def get_tokens():
    token_counts = {}
    longest_token_set = 0
    file_locations = ["test", "train", "val"]
    for file_location in file_locations:
        for file in os.scandir(file_location):
            if file.name.endswith("truth.txt"):
                continue
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


# returns the tweets of an author
def fetch_author_tweets(test, validation):
    author_tweets = {}
    if test:
        if validation:
            file_name = "val"
        else:
            file_name = "test"
    else:
        file_name = "train"
    for file in os.scandir(file_name):
        if file.name.endswith("truth.txt"):
            continue
        author = file.name.replace(".xml", "")
        file = open(file.path, encoding="utf8")
        author_tweets[author] = file.read()
        parser.data = []
        parser.feed(author_tweets[author])
        author_tweets[author] = parser.data
    return author_tweets


# returns the tweets of an author for the independence test
def fetch_author_tweets_ind(social, review):
    author_tweets = {}
    if social:
        file_name = "ind/social"
    else:
        if review:
            file_name = "ind/review"
        else:
            file_name = "ind/blog"
    for file in os.scandir(file_name):
        if file.name.endswith("truth.txt"):
            continue
        author = file.name.replace(".xml", "")
        file = open(file.path, encoding="utf8")
        author_tweets[author] = file.read()
        parser.data = []
        parser.feed(author_tweets[author])
        clean = re.compile('<.*?>')
        tweets = []
        for tweet in parser.data:
            tweets.append(re.sub(clean, '', tweet))
        author_tweets[author] = tweets
    return author_tweets


FLAGS = re.MULTILINE | re.DOTALL


# handles preprocessing of hashtags
def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = " <hashtag> {} <allcaps> ".format(hashtag_body)
    else:
        result = " ".join([" <hashtag> "] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
    return result


# handles preprocessing of allcaps
def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps>"


# handles preprocessing of tweets
def preprocess(text):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " <url> ")
    text = re_sub(r"/", " / ")
    text = re_sub(r"@\w+", " <user> ")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), " <smile> ")
    text = re_sub(r"{}{}p+".format(eyes, nose), " <lolface> ")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), " <sadface> ")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), " <neutralface> ")
    text = re_sub(r"<3", " <heart> ")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", " <number> ")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat> ")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong> ")
    text = re_sub(r"([A-Z]){2,}", allcaps)
    return text.lower()


# tokenizes the text
def tokenize(text):
    text = preprocess(text)
    tokens = []
    current_token = ""
    previous_was_alnum = False
    for char in text:
        if not char.isalnum():
            if (current_token != "" and char != "<" and char != ">") and (
                    previous_was_alnum or current_token[len(current_token) - 1] != char):
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


# returns the tdidf feature set for the requested dataset
def fetch_author_tweets_tokens(test, validation):
    author_tweets = fetch_author_tweets(test, validation)
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
    # generate idf portion of tfidf
    idf = np.zeros(len(token_numbering.keys()))
    total = 0
    for author in ordered_authors:
        total += 1
        author_set = np.zeros(len(token_numbering.keys()))
        for tweets in author_tweets_tokens[author]:
            for token in tweets:
                if token in token_numbering and author_set[token_numbering[token]] == 0:
                    author_set[token_numbering[token]] += 1
                    idf[token_numbering[token]] += 1
    for i in range(len(idf)):
        if idf[i] != 0:
            idf[i] = np.log(total / idf[i])
    # generate tfidf, average tweet length and average word length
    tensors = []
    for author in ordered_authors:
        length = 0
        tensor = np.zeros(len(token_numbering.keys()) + 2)
        number_of_tweets = 0
        number_of_tokens = 0
        token_length_sum = 0
        for tweets in author_tweets_tokens[author]:
            number_of_tweets += 1
            for token in tweets:
                if token in token_numbering:
                    number_of_tokens += 1
                    token_length_sum += len(token)
                    tensor[token_numbering[token]] += 1
                    length += 1
        for i in range(len(tensor) - 2):
            if tensor[i] != 0:
                tensor[i] = tensor[i] / length * idf[i]
        tensor[len(tensor) - 2] = number_of_tokens / number_of_tweets
        tensor[len(tensor) - 1] = token_length_sum / number_of_tokens
        tensors.append(tensor)
    return torch.Tensor(tensors)


# returns the tdidf feature set for the requested independence test
def fetch_author_tweets_tokens_ind(social, review):
    author_tweets = fetch_author_tweets_ind(social, review)
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
    # generate idf portion of tfidf
    idf = np.zeros(len(token_numbering.keys()))
    total = 0
    for author in ordered_authors:
        total += 1
        author_set = np.zeros(len(token_numbering.keys()))
        for tweets in author_tweets_tokens[author]:
            for token in tweets:
                if token in token_numbering and author_set[token_numbering[token]] == 0:
                    author_set[token_numbering[token]] += 1
                    idf[token_numbering[token]] += 1
    for i in range(len(idf)):
        if idf[i] != 0:
            idf[i] = np.log(total / idf[i])
    # generate tfidf, average tweet length and average word length
    tensors = []
    for author in ordered_authors:
        length = 0
        tensor = np.zeros(len(token_numbering.keys()) + 2)
        number_of_tweets = 0
        number_of_tokens = 0
        token_length_sum = 0
        for tweets in author_tweets_tokens[author]:
            number_of_tweets += 1
            for token in tweets:
                if token in token_numbering:
                    number_of_tokens += 1
                    token_length_sum += len(token)
                    tensor[token_numbering[token]] += 1
                    length += 1
        for i in range(len(tensor) - 2):
            if tensor[i] != 0:
                tensor[i] = tensor[i] / length * idf[i]
        tensor[len(tensor) - 2] = number_of_tokens / number_of_tweets
        if number_of_tokens != 0:
            tensor[len(tensor) - 1] = token_length_sum / number_of_tokens
        tensors.append(tensor)
    return torch.Tensor(tensors)


#generates the glove vectors needed for all the datasets

Word2VecModel = loadGloveModel("glove.twitter.27B.200d.txt")

AUTHOR_TWEETS = (fetch_author_tweets(False, False), fetch_author_tweets(True, False), fetch_author_tweets(True, True))
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
longest_tweet_length = 0
longest_number_of_tweets = 0
for i, author_tweets in enumerate(AUTHOR_TWEETS_TOKENS):
    token_tensors = {}
    for author in ORDERED_AUTHORS[i]:
        current_number_of_tweets = 0
        for tweets in author_tweets[author]:
            current_number_of_tweets += 1
            current_tweet_length = 0
            for token in tweets:
                if token.lower() in Word2VecModel:
                    current_tweet_length += 1
                    token_tensors[token] = Word2VecModel[token.lower()]
            if current_tweet_length > longest_tweet_length:
                longest_tweet_length = current_tweet_length
        if current_number_of_tweets > longest_number_of_tweets:
            longest_number_of_tweets = current_number_of_tweets
    token_tensors[" "] = np.zeros(200)
    TOKEN_TENSORS.append(token_tensors)


AUTHOR_TWEETS_IND = (fetch_author_tweets_ind(True, False), fetch_author_tweets_ind(False, True), fetch_author_tweets_ind(False, False))
ORDERED_AUTHORS_IND = []
AUTHOR_TWEETS_TOKENS_IND = []
for author_tweets in AUTHOR_TWEETS_IND:
    author_tweets_tokens = {}
    for author in author_tweets:
        tweets = author_tweets[author]
        tweets_tokens = []
        for tweet in tweets:
            tweet_tokens = tokenize(tweet)
            tweets_tokens.append(tweet_tokens)
        author_tweets_tokens[author] = tweets_tokens
    AUTHOR_TWEETS_TOKENS_IND.append(author_tweets_tokens)
    ORDERED_AUTHORS_IND.append(sorted(author_tweets_tokens.keys()))

token_numbering, longest_token_sequence = get_tokens()

TOKEN_TENSORS_IND = []
for i, author_tweets in enumerate(AUTHOR_TWEETS_TOKENS):
    token_tensors = {}
    for author in ORDERED_AUTHORS[i]:
        current_number_of_tweets = 0
        for tweets in author_tweets[author]:
            current_number_of_tweets += 1
            current_tweet_length = 0
            for token in tweets:
                if token.lower() in Word2VecModel:
                    current_tweet_length += 1
                    token_tensors[token] = Word2VecModel[token.lower()]
    token_tensors[" "] = np.zeros(200)
    TOKEN_TENSORS_IND.append(token_tensors)


#returns the word2vec model
def fetch_word2vec_model():
    return Word2VecModel


#fetch the GloVe feature set
def fetch_author_tweets_tokens_ordered(test, validation):
    authors = ORDERED_AUTHORS[test + validation]
    tensors = []
    sum_tweet_length = 0
    number_of_tweets = 0
    author_tweet_numbers = np.zeros([len(authors), 3])
    for j, author in enumerate(authors):
        author_number_of_tweets = 0
        tensor = []
        author_tweets_tokens = AUTHOR_TWEETS_TOKENS[test + validation]
        token_tensors = TOKEN_TENSORS[test + validation]
        for tweets in author_tweets_tokens[author]:
            tweet = []
            for token in tweets:
                if token.lower() in token_tensors:
                    tweet.append(token_tensors[token.lower()])
            i = len(tweet)
            if i != 0:
                sum_tweet_length += i
                number_of_tweets += 1
                author_number_of_tweets += 1
            while i < longest_tweet_length:
                tweet.append(token_tensors[" "])
                i += 1
            tensor.append(tweet)
        tweet = []
        i = len(tweet)
        while i < longest_tweet_length:
            tweet.append(token_tensors[" "])
            i += 1
        i = len(tensor)
        while i < longest_number_of_tweets:
            tensor.append(tweet)
            i += 1
        tensors.append(tensor)
        author_tweet_numbers[j][0] = author_number_of_tweets
    return torch.Tensor(tensors), author_tweet_numbers


#fetch the GloVe feature set for ta particular independence test
def fetch_author_tweets_tokens_ordered_ind(social, review):
    authors = ORDERED_AUTHORS_IND[int(review) + 2* int(not social and not review)]
    tensors = []
    author_tweet_numbers = np.zeros([len(authors), 3])
    for j, author in enumerate(authors):
        author_number_of_tweets = 0
        tensor = []
        author_tweets_tokens = AUTHOR_TWEETS_TOKENS_IND[int(review) + 2 * int(not social and not review)]
        token_tensors = TOKEN_TENSORS_IND[int(review) + 2* int(not social and not review)]
        for tweets in author_tweets_tokens[author]:
            tweet = []
            for token in tweets:
                if token.lower() in token_tensors:
                    tweet.append(token_tensors[token.lower()])
                if len(tweet) == longest_tweet_length:
                    tensor.append(tweet)
                    author_number_of_tweets += 1
                    tweet = []
            i = len(tweet)
            if i != 0:
                author_number_of_tweets += 1
            while i < longest_tweet_length:
                tweet.append(token_tensors[" "])
                i += 1
            tensor.append(tweet)
        tweet = []
        i = len(tweet)
        while i < longest_tweet_length:
            tweet.append(token_tensors[" "])
            i += 1
        i = len(tensor)
        while i < longest_number_of_tweets:
            tensor.append(tweet)
            i += 1
        tensors.append(tensor)
        author_tweet_numbers[j][0] = author_number_of_tweets
    return tensors, author_tweet_numbers


def fetch_tweets_tokens_ordered_with_truths(Test, Validation):
    tweet_tokens, tweet_numbers = fetch_author_tweets_tokens_ordered(Test, Validation)
    tweet_tokens_truths = fetch_author_truths(Test, Validation)
    age_truths = tweet_tokens_truths[0].numpy()
    gender_truths = tweet_tokens_truths[1].numpy()
    age_tweet_truths = []
    gender_tweet_truths = []
    tweets = []
    for i, tweet_set in enumerate(tweet_tokens):
        tweet_numbers[i][1] = age_truths[i]
        tweet_numbers[i][2] = gender_truths[i]
        for tweet in tweet_set:
            tokens = tweet.numpy()
            if tweet[0][0] != 0:
                tweets.append(tokens)
                age_tweet_truths.append(int(age_truths[i]))
                gender_tweet_truths.append(int(gender_truths[i]))
    toReturn = torch.Tensor(tweets)
    return toReturn, torch.tensor(age_tweet_truths), torch.tensor(gender_tweet_truths), tweet_numbers


def fetch_tweets_tokens_ordered_with_truths_ind(social, review):
    tweet_tokens, tweet_numbers = fetch_author_tweets_tokens_ordered_ind(social, review)
    tweet_tokens_truths = fetch_author_truths_ind(social, review)
    age_truths = tweet_tokens_truths[0].numpy()
    gender_truths = tweet_tokens_truths[1].numpy()
    age_tweet_truths = []
    gender_tweet_truths = []
    tweets = []
    for i, tweet_set in enumerate(tweet_tokens):
        tweet_numbers[i][1] = age_truths[i]
        tweet_numbers[i][2] = gender_truths[i]
        for tweet in tweet_set:
            tokens = tweet
            if tweet[0][0] != 0:
                tweets.append(tokens)
                age_tweet_truths.append(int(age_truths[i]))
                gender_tweet_truths.append(int(gender_truths[i]))
    toReturn = torch.Tensor(tweets)
    return toReturn, torch.tensor(age_tweet_truths), torch.tensor(gender_tweet_truths), tweet_numbers


def print_age_data(data, length):
    ageCat18 = len(torch.nonzero(data == 0))
    ageCat25 = len(torch.nonzero(data == 1))
    ageCat35 = len(torch.nonzero(data == 2))
    ageCat55 = len(torch.nonzero(data == 3))
    print("Aged 18-24: ", ageCat18 / length * 100)
    print("Aged 25-34: ", ageCat25 / length * 100)
    print("Aged 35-54: ", ageCat35 / length * 100)
    print("Aged 55+: ", ageCat55 / length * 100)


def print_gender_data(data, length):
    genderCatMale = len(torch.nonzero(data == 1))
    genderCatFemale = len(torch.nonzero(data == 0))
    print("Gender Male: ", genderCatMale / length * 100)
    print("Gender Female: ", genderCatFemale / length * 100)


def print_class_stats():
    print("TRAIN")
    train = fetch_author_truths(False, False)
    train_age = train[0]
    train_number = len(train_age)
    print(train_number)
    train_gender = train[1]
    print_age_data(train_age, train_number)
    print_gender_data(train_gender, train_number)
    print("VAL")
    val = fetch_author_truths(True, True)
    val_age = val[0]
    val_number = len(val_age)
    print(val_number)
    val_gender = val[1]
    print_age_data(val_age, val_number)
    print_gender_data(val_gender, val_number)
    print("TEST")
    test = fetch_author_truths(True, False)
    test_age = test[0]
    test_number = len(test_age)
    print(test_number)
    test_gender = test[1]
    print_age_data(test_age, test_number)
    print_gender_data(test_gender, test_number)


def print_tweet_stats():
    print("TRAIN")
    train = fetch_tweets_tokens_ordered_with_truths(False, False)
    train_age = train[1]
    train_number = len(train_age)
    print(train_number)
    train_gender = train[2]
    print_age_data(train_age, train_number)
    print_gender_data(train_gender, train_number)
    print("VAL")
    val = fetch_tweets_tokens_ordered_with_truths(True, True)
    val_age = val[1]
    val_number = len(val_age)
    print(val_number)
    val_gender = val[2]
    print_age_data(val_age, val_number)
    print_gender_data(val_gender, val_number)
    print("TEST")
    test = fetch_tweets_tokens_ordered_with_truths(True, False)
    test_age = test[1]
    test_number = len(test_age)
    print(test_number)
    test_gender = test[2]
    print_age_data(test_age, test_number)
    print_gender_data(test_gender, test_number)
