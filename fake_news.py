import pandas as pd
import time

from typing import Tuple
from nltk.corpus import stopwords
from pandarallel import pandarallel

from plots import plot_wordclouds, plot_title_lengths
from preprocessing import get_average_string_length, strip_non_letters, tokenize_column, calculate_bigrams_freqdist, \
    prepare_dataframes, remove_stopwords_from_df, vectorize_dataframes
from classification import train_evaluate_classifier
from sklearn.preprocessing import MinMaxScaler


def load_train_test_data(fake_path: str, true_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    print(f"Fake set length: {len(df_fake)}, true set length: {len(df_true)}")

    return df_fake, df_true


def main():
    fake_path = 'data/Fake.csv'
    true_path = 'data/True.csv'

    # Load data
    df_fake, df_true = load_train_test_data(fake_path, true_path)

    # Plot wordcloud
    plot_wordclouds(df_fake, df_true)

    # Cleanup non letters
    df_fake = strip_non_letters(df_fake, ['title', 'text'])
    df_true = strip_non_letters(df_true, ['title', 'text'])

    # Get lengths of titles and texts
    print(f"Average character length in fake titles is {get_average_string_length(df_fake, 'title')}")
    print(f"Average character length in fake texts is {get_average_string_length(df_fake, 'text')}")
    print(f"Average character length in true titles is {get_average_string_length(df_true, 'title')}")
    print(f"Average character length in fake titles is {get_average_string_length(df_true, 'text')}", '\n')

    # Plot title lengths for fake and true titles/texts
    plot_title_lengths(df_fake, df_true)

    # Remove stopwords and do the same
    eng_stopwords = set(stopwords.words('english'))
    for col_name in ['title', 'text']:
        df_fake = remove_stopwords_from_df(df_fake, col_name, stopwords_set=eng_stopwords)
        df_true = remove_stopwords_from_df(df_true, col_name, stopwords_set=eng_stopwords)

    print(f"Average character length in filtered fake titles is {get_average_string_length(df_fake, 'title')}")
    print(f"Average character length in filtered fake texts is {get_average_string_length(df_fake, 'text')}")
    print(f"Average character length in filtered true titles is {get_average_string_length(df_true, 'title')}")
    print(f"Average character length in filtered fake titles is {get_average_string_length(df_true, 'text')}",
          '\n')

    plot_title_lengths(df_fake, df_true)

    # Word tokenize titles and texts
    print("Tokenizing corpus...")
    pandarallel.initialize(progress_bar=True)
    start_time = time.time()
    for column, tokenized_column in zip(['title', 'text'], ['tokenized_title', 'tokenized_text']):
        df_fake = tokenize_column(df_fake, column, tokenized_column)
        df_true = tokenize_column(df_true, column, tokenized_column)

    print(f"Tokenized corpus in {time.time() - start_time} seconds", '\n')

    # Calculate bigram frequency and show the 10 most common ones
    print("Calculating bigrams and frequency distributions...")
    for column in ['title', 'text']:
        fake_fdist = calculate_bigrams_freqdist(df_fake, column)
        print(f"10 common fake {column} bigrams: {fake_fdist.most_common(n=10)}")

        true_fdist = calculate_bigrams_freqdist(df_true, column)
        print(f"10 common true {column} bigrams: {true_fdist.most_common(n=10)}")

    # Prepare dataframes for classification
    df_train, df_test = prepare_dataframes(df_fake, df_true, test_size=0.2)

    # Various vectorization methods
    corpus_dict = {"bow": {"train": [], "test": []}, "tfidf": {"train": [], "test": []},
                   "doc2vec": {"train": [], "test": []}}

    for vectorizer in ['bow', 'tfidf', 'doc2vec']:
        start_time = time.time()
        print(f"Vectorizing corpus with {vectorizer} vectorization method...")
        X_train, X_test = vectorize_dataframes(df_train, df_test, vectorizer=vectorizer, n_words=5000)
        corpus_dict[vectorizer]["train"] = X_train
        corpus_dict[vectorizer]["test"] = X_test
        print(f"Vectorized with {vectorizer} in {time.time() - start_time} seconds")

    # Train and evaluate classifiers on all corpora
    score_dict = {"bow": {"svm": {}, "naive_bayes": {}, "random_forest": {}, "logistic_regression": {}},
                  "tfidf": {"svm": {}, "naive_bayes": {}, "random_forest": {}, "logistic_regression": {}},
                  "doc2vec": {"svm": {}, "naive_bayes": {}, "random_forest": {}, "logistic_regression": {}}}

    for corpus_type, values in corpus_dict.items():
        # Scale data for Doc2Vec
        if corpus_type == "doc2vec":
            scaler = MinMaxScaler()
            values["train"] = scaler.fit_transform(values["train"])
            values["test"] = scaler.transform(values["test"])
        for classifier in ['svm', 'naive_bayes', 'random_forest', 'logistic_regression']:
            print(f"Training {corpus_type} {classifier} classifier...")
            start_time = time.time()

            score_dict[corpus_type][classifier] = train_evaluate_classifier(classifier, x_train=values["train"],
                                                                            y_train=df_train["label"],
                                                                            x_test=values["test"],
                                                                            y_test=df_test["label"])
            print(f"Finished training and evaluating {corpus_type} {classifier} in {time.time() - start_time} seconds")

    print(f"Final results: {score_dict}")


if __name__ == "__main__":
    main()
