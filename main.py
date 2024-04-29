import os
import sys
import numpy as np
import csv
import multiprocessing
from functools import partial
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import spacy
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from textblob import TextBlob
from sklearn import cluster
import pandas as pd


def leastNumber(*data):
    if len(data) == 0:
        return 0
    return min(data)


"""
FIRST STEP
Converting the dataset from JSON to CSV
"""
# with open('GoogleReviewNY.json', encoding='utf-8') as inputfile:
#     df = pd.read_json(inputfile, lines=True)
#
# df.to_csv('reviews_in_ny.csv', encoding='utf-8', index=False)

"""
SECOND STEP
Converting the dataset from CSV to TSV
"""

# input = os.path.join(os.path.dirname(sys.argv[0]), "reviews_in_ny.csv")
# output = os.path.join(os.path.dirname(sys.argv[0]), "reviews_in_ny.tsv")
#
# with open(input, 'r', newline='') as csv_file, open(output, 'w', newline='') as tsv_file:
#     csv_reader = csv.reader(csv_file)
#     tsv_writer = csv.writer(tsv_file, delimiter='\t')
#
#     for row in csv_reader:
#         tsv_writer.writerow(row)


"""
THIRD STEP
replacing blank with "null" elements
remove lines with insufficient columns that are less than 8 columns 
removes duplicated lines
"""

# input = os.path.join(os.path.dirname(sys.argv[0]), "reviews_in_ny.tsv")
# output = os.path.join(os.path.dirname(sys.argv[0]), "reviews_in_ny_modified.tsv")
#
# with open(input, 'r', newline='') as file_reader, open(output, 'w', newline='') as file_writer:
#     reader = csv.reader(file_reader, delimiter='\t')
#     writer = csv.writer(file_writer, delimiter='\t')
#     written_lines = set()
#
#     for line in file_reader:
#         line = line.rstrip()
#         line = line.split("\t")
#         if len(line) < 8:
#             print("Skipping line with insufficient columns:", line)
#             continue
#
#         key = tuple(line)
#         if key in written_lines:
#             print("Skipping duplicate line:", key)
#             continue
#         written_lines.add(key)
#
#         update_row = []
#         for cell in line:
#             if cell == "":
#                 update_row.append("null")
#             else:
#                 update_row.append(cell)
#         writer.writerow(update_row)


"""
FOURTH STEP
Remove lines that don't have text because I will be feature engineering solely with it.
"""
# input = os.path.join(os.path.dirname(sys.argv[0]), "reviews_in_ny_modified.tsv")
# output = os.path.join(os.path.dirname(sys.argv[0]), "reviews_in_ny_remove_text_null.tsv")
#
# with open(input, 'r', newline='') as text_with_null, open(output, 'w', newline='') as text_without_null:
#
#     #Create reader and writer objects
#     null_reader = csv.reader(text_with_null, delimiter="\t")
#     without_null_writer = csv.writer(text_without_null, delimiter="\t")
#
#     for line in null_reader:
#         if line[4] == "null":
#             print("Skipping line because it has no text value", line)
#             continue
#         without_null_writer.writerow(line)


"""
FIFTH STEP

Create a new column that identify if it is positive or negative review. 1 is positive and 0 is negative
"""
# input = os.path.join(os.path.dirname(sys.argv[0]), "reviews_in_ny_remove_text_null.tsv")
# output = os.path.join(os.path.dirname(sys.argv[0]), "reviews_in_ny_modified.tsv")
#
#
# with open(input, 'r', newline= '') as before, open(output,'w', newline='') as after:
#     b = csv.reader(before, delimiter="\t")
#     a = csv.writer(after, delimiter="\t")
#
#     header = next(b)
#     header.append("Positive/Negative")
#     a.writerow(header)
#
#     count = 0
#     for line in b:
#         temp = line[3]
#         count += 1
#         print(f"Starting: {count} --- Leftover: {8895335 - count}")
#         print(temp)
#         if temp in ["1", "2", "3"]:
#             value = "0"
#         else:
#             value = "1"
#         print(value)
#         line.append(value)
#         print("Completed\n\n\n")
#         a.writerow(line)


"""
SIXTH STEP
remove extra data from the classes to make them equal numbers of data as the class with the shortest data
"""
# input = os.path.join(os.path.dirname(sys.argv[0]), "reviews_in_ny_modified.tsv")
# output = os.path.join(os.path.dirname(sys.argv[0]), "reviews_in_ny_f.tsv")
#
#
# with open(input, 'r', newline='') as unbalanced:
#     count0 = 0
#     count1 = 0
#     ub_reader = csv.reader(unbalanced, delimiter="\t")
#     header = next(ub_reader)
#     for row in ub_reader:
#         if row[8] == "0":
#             count0 += 1
#         else:
#             count1 += 1
#
# least_number = leastNumber(count0, count1)
#
# print("BEFORE:")
# print(f'0: {count0}\n1: {count1}')
#
# print(f"least number is {least_number}")
#
# with open(input, 'r', newline='') as unbalanced, open(output, 'w', newline='') as balanced:
#     ub = csv.reader(unbalanced, delimiter="\t")
#     b = csv.writer(balanced, delimiter="\t")
#
#     count0 = 0
#     count1 = 0
#
#     b.writerow(header)
#     next(ub)
#
#     for line in ub:
#         rate = line[8]
#         if rate == "0" and count0 < least_number:
#             b.writerow(line)
#             count0 += 1
#         elif rate == "1" and count1 < least_number:
#             b.writerow(line)
#             count1 += 1
#
# print("AFTER:")
# print(f'0: {count0}\n1: {count1}')

"""
SEVENTH STEP ------------------
take portion of the dataset so it is faster to train and test the data for developing a model
"""

# input = os.path.join(os.path.dirname(sys.argv[0]), "reviews_in_ny_f.tsv")
# output = os.path.join(os.path.dirname(sys.argv[0]), "reviews_in_ny_s.tsv")
#
# '''
#     OLD CODE
#
#     # #Create reader and writer objects
#     # full_reader = csv.reader(full, delimiter="\t")
#     # short_writer = csv.writer(short, delimiter="\t")
#     #
#     # rows_written = 0
#     # for i, row in enumerate(full_reader):
#     #     if rows_written < 3000:
#     #
#     #         short_writer.writerow(row)
#     #         rows_written += 1
#     #     else:
#     #         break
# '''
#
# with open(input, 'r', newline='') as full:
#     count0 = 0
#     count1 = 0
#     ub_reader = csv.reader(full, delimiter="\t")
#     header = next(ub_reader)
#     for row in ub_reader:
#         if row[8] == "0":
#             count0 += 1
#         else:
#             count1 += 1
#
# least_number = 3000
#
# print("BEFORE:")
# print(f'0: {count0}\n1: {count1}')
#
# print(f"least number is {least_number}")
#
# with open(input, 'r', newline='') as full, open(output, 'w', newline='') as short:
#     ub = csv.reader(full, delimiter="\t")
#     b = csv.writer(short, delimiter="\t")
#
#     count0 = 0
#     count1 = 0
#
#     b.writerow(header)
#     next(ub)
#
#     for line in ub:
#         rate = line[8]
#         if rate == "0" and count0 < least_number:
#             b.writerow(line)
#             count0 += 1
#         elif rate == "1" and count1 < least_number:
#             b.writerow(line)
#             count1 += 1
#
# print("AFTER:")
# print(f'0: {count0}\n1: {count1}')

"""
EIGHTH STEP
have fun building features and training/testing the data :)
"""

nlp = spacy.load("en_core_web_sm")


def textLength(text):
    return len(text)


# phrase = ("Hello, this is Awais. It is nice to meet you. How is your day going so far. I am so shocked to hear that "
#           "you will be moving 500 miles away from here. We will for sure miss you a lot. Anyway, have you heard the "
#           "rumors about Ryan Garcia. His instagram posts are really concerning.")
#
# phrase_token = textLength(phrase)
#
# phrase_bi = list(nltk.bigrams(phrase_token))
#
# print(phrase_bi)
#
# phrase_tri = list(nltk.trigrams(phrase_token))
#
# print(phrase_tri)
#
# pst = PorterStemmer()
#
# # for word in phrase_token:
# #     result = pst.stem(word)
# #     print(result)
#
# punctuation = re.compile(r'[-.?!,:;()|0-9]')
# post_punctuation = []
#
# for word in phrase_token:
#     word = punctuation.sub("",word)
#     if len(word)>0:
#         post_punctuation.append(word)
# print(post_punctuation)
#
# english = stopwords.words('english')
# post_stopwords = []
#
# for word in post_punctuation:
#     if word in english:
#         continue
#     else:
#         post_stopwords.append(word)
# print(post_stopwords)
#
# analysis = TextBlob(phrase)
# polarity = analysis.sentiment.polarity
#
# print(phrase)
# print(polarity)
#
def emotionScore(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    # print(polarity)
    return polarity


def trigramAvgLength(text):
    trigramTotal = 0
    trigramAmt = 0
    trigram = []
    for token in text:
        if len(trigram) == 3:
            tmpTotal = 0
            for tkn in trigram:
                tmpTotal = tmpTotal + len(tkn)
            trigramTotal = trigramTotal + tmpTotal
            trigram = []
            trigramAmt = trigramAmt + 1
        else:
            trigram.append(token)

    if len(trigram) == 3:
        tmpTotal = sum(len(tkn) for tkn in trigram)
        trigramTotal += tmpTotal
        trigramAmt += 1

    return trigramTotal / trigramAmt if trigramAmt != 0 else 0


def countNouns(text):
    doc = nlp(text)
    nounTotal = 0
    for token in doc:
        if token.tag_ == "NN" or token.tag_ == "NNP":
            nounTotal = nounTotal + 1
    return nounTotal


def countVerbs(text):
    doc = nlp(text)
    verbTotal = 0
    for token in doc:
        if token.tag_ == "VBZ" or token.tag_ == "VBG":
            verbTotal = verbTotal + 1
    return verbTotal


def countAdj(text):
    doc = nlp(text)
    adjTotal = 0
    for token in doc:
        if token.tag_ == "ADJ":
            adjTotal = adjTotal + 1
    return adjTotal


def countAdv(text):
    doc = nlp(text)
    advTotal = 0
    for token in doc:
        if token.tag_ == "ADV":
            advTotal = advTotal + 1
    return advTotal


def startsWithNumber(text):
    beginningNumber = 0
    tokens = text.split()

    if tokens[0].isnumeric() == True:
        beginningNumber = 1

    return beginningNumber


def getWordCount(text):
    tokens = text.split()
    return len(tokens)


def countStopwords(text):
    english = stopwords.words('english')
    tokens = text.split()

    stopwordCount = 0
    for token in tokens:
        for stopword in english:
            if token.lower() == stopword.lower():
                stopwordCount = stopwordCount + 1
    return stopwordCount


def countPronounWords(text):
    pronouns = []
    with open("pronounwords.txt") as file:
        for line in file:
            ln = line.rstrip()
            pronouns.append(ln)

    tokens = text.split()
    pronounCount = 0
    for token in tokens:
        # print(token)
        for pronoun in pronouns:
            # print(pronoun)
            if token.lower() == pronoun.lower():
                pronounCount = pronounCount + 1
    return pronounCount


def textFeatures(texts):
    texts_features = []
    counter = 1
    for text in texts:
        f1 = textLength(text)
        f2 = trigramAvgLength(text)
        f3 = countNouns(text)
        f4 = countVerbs(text)
        f5 = countAdj(text)
        f6 = countAdv(text)
        f7 = emotionScore(text)
        f8 = startsWithNumber(text)
        f9 = getWordCount(text)
        f10 = countStopwords(text)
        f11 = countPronounWords(text)
        text_features = [f1, f2, f7, f8]
        texts_features.append(text_features)
        print(f'{counter} - {3212838 - counter} {text}')
        counter += 1
    return texts_features


if __name__ == '__main__':

    texts = []
    classLabels = []

    firstLine = True

    with open("reviews_in_ny_f.tsv") as file:
        for line in file:
            line = line.split("\t")
            if firstLine == True:
                firstLine = False
                continue
            texts.append(line[4])
            review = line[8].removesuffix("\n") # not sure why it added "\n" in the first place
            classLabels.append(review)

    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)

    chunk_size = len(texts) // num_processes
    text_chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

    features_chunks = pool.map(textFeatures, text_chunks)

    texts_features = [feature for features_chunk in features_chunks for feature in features_chunk]

    pool.close()
    pool.join()

    X_train, X_test, y_train, y_test = train_test_split(texts_features, classLabels, test_size=0.2)

    print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)

    classifierRndForest = RandomForestClassifier()
    classifierRndForest.fit(X_train, y_train)
    rndForestTestPred = classifierRndForest.predict(X_test)
    rndForestAccuracy = np.mean(rndForestTestPred == y_test)
    print("Random Forest Test set score: {:.2f}".format(rndForestAccuracy))

    classifierDTree = DecisionTreeClassifier()
    classifierDTree.fit(X_train, y_train)
    dTreeTestPred = classifierDTree.predict(X_test)
    dTreeAccuracy = np.mean(dTreeTestPred == y_test)
    print("Decision Tree Test set score: {:.2f}".format(dTreeAccuracy))

    classifierNB = GaussianNB()
    classifierNB.fit(X_train, y_train)
    nbTestPred = classifierNB.predict(X_test)
    nbAccuracy = np.mean(nbTestPred == y_test)
    print("Gaussian NB Test set score: {:.2f}".format(nbAccuracy))

    classifierKNN = KNeighborsClassifier(n_neighbors=3)
    classifierKNN.fit(X_train, y_train)
    knnTestPred = classifierKNN.predict(X_test)
    knnAccuracy = np.mean(knnTestPred == y_test)
    print("K-Nearest Neighbour Test set score: {:.2f}".format(knnAccuracy))

    classifierSVM = svm.LinearSVC()
    classifierSVM.fit(X_train, y_train)
    svmTestPred = classifierSVM.predict(X_test)
    svmAccuracy = np.mean(svmTestPred == y_test)
    print("SVM Test set score: {:.2f}".format(svmAccuracy))


    def plot_confusion_matrix(y_true, y_pred, classLabel, color, title):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap=color, cbar=False, xticklabels=classLabel, yticklabels=classLabel)
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()


    classifiers = ["Random Forest", "Decision Tree", "Gaussian NB", "K-Nearest Neighbor", "SVM"]
    accuracies = [rndForestAccuracy, dTreeAccuracy, nbAccuracy, knnAccuracy, svmAccuracy]

    plt.figure(figsize=(10, 6))
    plt.bar(classifiers, accuracies, color=['blue', 'green', 'orange', 'red', 'purple'])
    plt.title('Classifier Test Set Accuracy')
    plt.ylabel('Accuracy')
    plt.show()

    class_labels = ['Negative', 'Positive']
    classifiers = [classifierRndForest, classifierDTree, classifierNB, classifierKNN, classifierSVM]
    for i, classifier in enumerate(classifiers):
        if i == 0:
            test_pred = rndForestTestPred
            title = 'Random Forest'
            color = "Greens"

        elif i == 1:
            test_pred = dTreeTestPred
            title = 'Decision Tree'
            color = "Blues"
        elif i == 2:
            test_pred = nbTestPred
            title = 'Gaussian NB'
            color = "Reds"
        elif i == 3:
            test_pred = knnTestPred
            title = 'K-Nearest Neighbor'
            color = "Oranges"
        elif i == 4:
            test_pred = svmTestPred
            title = 'SVM'
            color = "Purples"

        plot_confusion_matrix(y_test, test_pred, class_labels, color, title)

