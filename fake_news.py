"""FAKE NEWS CLASSIFICATION SYSTEM"""
import pandas as pd
import re
import matplotlib.pyplot as plt
import nltk
import numpy as np
from wordcloud import WordCloud
from gensim.parsing.preprocessing import remove_stopwords
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from collections import Counter

# First we'll analyze the dataset, only keeping the title and the text, and we'll remove punctuation and numbers
df_fake = pd.read_csv('data/Fake.csv')
df_true = pd.read_csv('data/True.csv')

print(len(df_true), len(df_fake))
print('Generating wordclouds...')
fake_cloud = WordCloud(background_color='white').generate("".join(title for title in df_fake['title']))
true_cloud = WordCloud(background_color='white').generate("".join(title for title in df_true['title']))
# Plot wordclouds
figure, axes = plt.subplots(nrows=1, ncols=2)
axes[0].set_title('Fake news')
axes[0].imshow(fake_cloud, interpolation='bilinear')
axes[1].set_title('True news')
axes[1].imshow(true_cloud, interpolation='bilinear')
figure.tight_layout()
del fake_cloud
del true_cloud


# Function that accepts a list of strings and only keeps the letters from the alphabet for each word
def list_preprocess(raw_list):
    new_list = []
    for text in raw_list:
        new_list.append(re.sub('[^a-zA-Z]+', ' ', text))
    return new_list


# Find average string length in titles/text for each category
fake_titles = list_preprocess(df_fake['title'].tolist())
print('Average character length in fake titles is: ', sum(map(len, fake_titles)) / float(len(fake_titles)))
true_titles = list_preprocess(df_true['title'].tolist())
print('Average character length in true titles is: ', sum(map(len, true_titles)) / float(len(true_titles)))
fake_text = list_preprocess(df_fake['text'].tolist())
print('Average character length in fake news is: ', sum(map(len, fake_text)) / float(len(fake_text)))
true_text = list_preprocess(df_true['text'].tolist())
print('Average character length in true news is: ', sum(map(len, true_text)) / float(len(true_text)))

# Plot title lengths
figure, axes = plt.subplots(nrows=2, ncols=2)
axes[0, 0].set_title('Fake title length distribution')
axes[0, 0].hist([len(title.split()) for title in fake_titles])
axes[0, 1].set_title('True title length distribution')
axes[0, 1].hist([len(title.split()) for title in true_titles])
axes[1, 0].set_title('Fake news length distribution')
axes[1, 0].hist([len(text.split()) for text in fake_text])
axes[1, 1].set_title('True news length distribution')
axes[1, 1].hist([len(text.split()) for text in true_text])

figure.tight_layout()
plt.show()

# Same but with removed stopwords
filtered_fake_titles = [remove_stopwords(title) for title in fake_titles]
filtered_true_titles = [remove_stopwords(title) for title in true_titles]
filtered_fake_text = [remove_stopwords(text) for text in fake_text]
filtered_true_text = [remove_stopwords(text) for text in true_text]

figure, axes = plt.subplots(nrows=2, ncols=2)
axes[0, 0].set_title('Fake title length distribution -- NO STOPWORDS')
axes[0, 0].hist([len(title.split()) for title in filtered_fake_titles])
axes[0, 1].set_title('True title length distribution -- NO STOPWORDS')
axes[0, 1].hist([len(title.split()) for title in filtered_true_titles])
axes[1, 0].set_title('Fake news length distribution -- NO STOPWORDS')
axes[1, 0].hist([len(text.split()) for text in filtered_fake_text])
axes[1, 1].set_title('True news length distribution -- NO STOPWORDS')
axes[1, 1].hist([len(text.split()) for text in filtered_true_text])

figure.tight_layout()
plt.show()

# Find bigram frequency in fake/real titles and texts (without the stopwords)
print('Calculating bigram frequencies (no stopwords)...')
fake_title_tokens = nltk.word_tokenize(' '.join(filtered_fake_titles))
true_title_tokens = nltk.word_tokenize(' '.join(filtered_true_titles))
fake_text_tokens = nltk.word_tokenize(' '.join(filtered_fake_text))
true_text_tokens = nltk.word_tokenize(' '.join(filtered_true_text))

fake_title_bigrams = nltk.bigrams(fake_title_tokens)
true_title_bigrams = nltk.bigrams(true_title_tokens)
fake_text_bigrams = nltk.bigrams(fake_text_tokens)
true_text_bigrams = nltk.bigrams(true_text_tokens)

fake_title_fdist = nltk.FreqDist(fake_title_bigrams)
true_title_fdist = nltk.FreqDist(true_title_bigrams)
fake_text_fdist = nltk.FreqDist(fake_text_bigrams)
true_text_fdist = nltk.FreqDist(true_text_bigrams)

print('Most common fake title bigrams: ', fake_title_fdist.most_common(n=10))
print('Most common true title bigrams: ', true_title_fdist.most_common(n=10))
print('Most common fake text bigrams: ', fake_text_fdist.most_common(n=10))
print('Most common true text bigrams: ', true_text_fdist.most_common(n=10))

# Select 10000 fake and true news to create a new csv. Add a label column to use for training/testing (1 for true
# news, 0 for fake). fake.csv has 23481 rows while true.csv has 21417.
df_fake['label'] = 0
df_true['label'] = 1
train_df = df_fake.head(n=10000).append(df_true.head(n=10000))
test_df = df_fake.tail(n=13481).append(df_true.tail(n=11417))
train_df.to_csv('data/train.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

# Classification for a start, first read the csvs
train_df = pd.read_csv('data/train.csv')[9500:10500]
test_df = pd.read_csv('data/test.csv')
# Combine title, text and subject in one column
train_df['text'] = train_df['title'] + ' ' + train_df['text'] + ' ' + train_df['subject']
test_df['text'] = test_df['title'] + ' ' + test_df['text'] + ' ' + test_df['subject']
# print('Example text: ', train_df['text'][0])
# Create bag of words and TF-IDF vector representations
print('Creating Bag of Words and TF-IDF representations...')
bow_vect = CountVectorizer(stop_words='english')
tfidf_vect = TfidfVectorizer(stop_words='english')
X_train_bow = bow_vect.fit_transform(train_df['text'])
X_train_tfidf = tfidf_vect.fit_transform(train_df['text'])
X_test_bow = bow_vect.transform(test_df['text'])
X_test_tfidf = tfidf_vect.transform(test_df['text'])
y_train = train_df['label']
y_test = test_df['label']
print(Counter(y_train), Counter(y_test))

# Create Word2Vec representation.
print('Training Word2Vec model (after tokenization)...')
tokenized_train_corpus = [nltk.word_tokenize(news) for news in train_df['text']]
tokenized_test_corpus = [nltk.word_tokenize(news) for news in test_df['text']]
# Learn word vectors from the corpus, dimensionality is 100
model = Word2Vec(tokenized_train_corpus, vector_size=100, window=5, min_count=5, workers=4)
model.train(tokenized_train_corpus, total_examples=len(tokenized_train_corpus), epochs=5)
X_train_w2v = []
print('Transforming train documents to w2v representation...')
news_counter = 0
for news in tokenized_train_corpus:
    if len(news) > 0:
        text = [word for word in news if word in model.wv.key_to_index]
    else:
        text = ['empty']
    news_counter += 1
    # Take the average of each vector
    w2v_news = np.mean(model.wv[text], axis=0)
    X_train_w2v.append(w2v_news)

# Sanity check and conversion to numpy array
print('Processed this number of articles: ', len(X_train_w2v))
X_train_w2v = np.array(X_train_w2v)
print('Train corpus shape after word2vec conversion', X_train_w2v.shape)
# Also transform the test set for usage later on
X_test_w2v = []
for news in tokenized_test_corpus:
    if len(news) > 0:
        text = [word for word in news if word in model.wv.key_to_index]
    else:
        text = ['empty']
    news_counter += 1
    # Take the average of each vector
    w2v_news = np.mean(model.wv[text], axis=0)
    X_test_w2v.append(w2v_news)

# Sanity check and conversion to numpy array
print('Processed this number of articles: ', len(X_test_w2v))
X_test_w2v = np.array(X_test_w2v)
print('Test corpus shape after word2vec conversion', X_test_w2v.shape)

# We'll start by evaluating how the Bag of Words model works across all 4 classifiers.
parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10], 'gamma': ('scale', 'auto')}
scoring = {'Accuracy': 'accuracy', 'F-Measure': 'f1'}
print('Training SVM classifier(BoW)')
svc = SVC()
svc_clf = GridSearchCV(svc, parameters, n_jobs=-1, scoring=scoring, verbose=3, refit='Accuracy')
svc_clf.fit(X_train_bow, y_train)
bow_svc_score = svc_clf.score(X_test_bow, y_test)
# Naive Bayes
print('Training Complement Naive Bayes classifier(BoW)')
bayes = ComplementNB()
bayes.fit(X_train_bow, y_train)
bow_bayes_acc = bayes.score(X_test_bow, y_test)
bow_bayes_f1 = f1_score(y_test, bayes.predict(X_test_bow))
# Random forest
print('Training Random Forest Classifier(BoW)')
forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=1)
forest.fit(X_train_bow, y_train)
bow_forest_acc = forest.score(X_test_bow, y_test)
bow_forest_f1 = f1_score(y_test, forest.predict(X_test_bow))
# Logistic Regression
print('Training logistic regression classifier(BoW)')
regr = LogisticRegression(n_jobs=-1)
regr.fit(X_train_bow, y_train)
bow_regr_acc = regr.score(X_test_bow, y_test)
bow_regr_f1 = f1_score(y_test, regr.predict(X_test_bow))

# Now TF-IDF
print('Training SVM classifier(TF-IDF)')
svc_clf.fit(X_train_tfidf, y_train)
tfidf_svc_score = svc_clf.score(X_test_tfidf, y_test)
# Naive Bayes
print('Training Gaussian Naive Bayes classifier(TF-IDF)')
bayes = GaussianNB()
bayes.fit(X_train_tfidf.toarray(), y_train)
tfidf_bayes_acc = bayes.score(X_test_tfidf.toarray(), y_test)
tfidf_bayes_f1 = f1_score(y_test, bayes.predict(X_test_tfidf.toarray()))
# Random forest
print('Training Random Forest Classifier(TF-IDF)')
forest.fit(X_train_tfidf, y_train)
tfidf_forest_acc = forest.score(X_test_tfidf, y_test)
tfidf_forest_f1 = f1_score(y_test, forest.predict(X_test_tfidf))
# Logistic Regression
print('Training logistic regression classifier(TF-IDF)')
regr.fit(X_train_tfidf, y_train)
tfidf_regr_acc = regr.score(X_test_tfidf, y_test)
tfidf_regr_f1 = f1_score(y_test, regr.predict(X_test_tfidf))

# Finally, Word2Vec
print('Training SVM classifier(Word2Vec)')
svc_clf.fit(X_train_w2v, y_train)
w2v_svc_score = svc_clf.score(X_test_w2v, y_test)
# Naive Bayes
print('Training Gaussian Naive Bayes classifier(Word2Vec)')
bayes.fit(X_train_w2v, y_train)
w2v_bayes_acc = bayes.score(X_test_w2v, y_test)
w2v_bayes_f1 = f1_score(y_test, bayes.predict(X_test_w2v))
# Random forest
print('Training Random Forest Classifier(Word2Vec)')
forest.fit(X_train_w2v, y_train)
w2v_forest_acc = forest.score(X_test_w2v, y_test)
w2v_forest_f1 = f1_score(y_test, forest.predict(X_test_w2v))
# Logistic Regression
print('Training logistic regression classifier(Word2Vec)')
regr.fit(X_train_w2v, y_train)
w2v_regr_acc = regr.score(X_test_w2v, y_test)
w2v_regr_f1 = f1_score(y_test, regr.predict(X_test_w2v))
print(y_test)
# Summarise the results
bow_dict = {'SVM: ': bow_svc_score, 'Naive Bayes accuracy: ': bow_bayes_acc, 'Naive Bayes f1: ': bow_bayes_f1,
            'Random Forest accuracy: ': bow_forest_acc, 'Random Forest f1: ': bow_forest_f1,
            'Logistic Regression accuracy:': bow_regr_acc, 'Logistic Regression f1:': bow_regr_f1}
tfidf_dict = {'SVM: ': tfidf_svc_score, 'Naive Bayes accuracy: ': tfidf_bayes_acc, 'Naive Bayes f1: ': tfidf_bayes_f1,
              'Random Forest accuracy: ': tfidf_forest_acc, 'Random Forest f1: ': tfidf_forest_f1,
              'Logistic Regression accuracy:': tfidf_regr_acc, 'Logistic Regression f1:': tfidf_regr_f1}
w2v_dict = {'SVM: ': w2v_svc_score, 'Naive Bayes accuracy: ': w2v_bayes_acc, 'Naive Bayes f1: ': w2v_bayes_f1,
            'Random Forest accuracy: ': w2v_forest_acc, 'Random Forest f1: ': w2v_forest_f1,
            'Logistic Regression accuracy:': w2v_regr_acc, 'Logistic Regression f1:': w2v_regr_f1}

print('Results for Bag of Words representation:')
print(bow_dict)
print('Results for TF-IDF representation:')
print(tfidf_dict)
print('Results for Word2Vec representation:')
print(w2v_dict)
