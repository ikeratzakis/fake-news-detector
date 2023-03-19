import pandas as pd

from typing import Tuple, List
from nltk import bigrams, FreqDist
from nltk.tokenize import word_tokenize

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def strip_non_letters(df: pd.DataFrame, col_names: List[str]) -> pd.DataFrame:
    """
    Strip all non letter characters for all pandas columns contained in a list
    :param df: Pandas dataframe
    :param col_names: The name of the column to perform cleaning on
    :return: DataFrame with the preprocessed columns
    """
    for col_name in col_names:
        df[col_name] = df[col_name].str.replace('\W', ' ')
        df[col_name] = df[col_name].str.replace('\d', '')
    return df


def get_average_string_length(df: pd.DataFrame, col_name: str) -> float:
    """
    Calculate average length of a column string in pandas DataFrame
    :param df: Pandas dataframe
    :param col_name: Column to apply function on
    :return:
    """
    return df[col_name].apply(len).mean()


def remove_stopwords_from_df(df: pd.DataFrame, col_name: str, stopwords_set: set) -> pd.DataFrame:
    df[col_name] = df[col_name].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords_set]))

    return df


def tokenize_column(df: pd.DataFrame, col_name: str, new_col_name: str) -> pd.DataFrame:
    """
    Perform word tokenization on a pandas column and return a dataframe with the tokenized column as new
    :param df: Pandas dataframe to apply function on
    :param col_name: Name of column to tokenize
    :param new_col_name: Name of the tokenized column
    :return: Pandas dataframe with the tokenized column
    """

    # Use parallel version of pandas apply for texts since they are large and can benefit from multiple cores
    if col_name == 'text':
        df[new_col_name] = df[col_name].parallel_apply(word_tokenize)
    else:
        df[new_col_name] = df[col_name].apply(word_tokenize)
    return df


def calculate_bigrams_freqdist(df: pd.DataFrame, col_name: str) -> FreqDist:
    """
    Calculate bigrams frequency distribution from a pandas column
    :param df: Pandas DataFrame to apply function on
    :param col_name: Name of column to compute bigrams for
    :return: Bigram frequency distribution
    """
    _bigrams = bigrams(df[col_name].values.tolist())
    freq_dist = FreqDist(_bigrams)
    return freq_dist


def prepare_dataframes(df_fake: pd.DataFrame, df_true: pd.DataFrame, test_size: float = 0.2) -> \
        Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare the dataframes for classification algorithms by merging them, randomly shuffling and splitting to train/test
    :param df_fake: Pandas dataframe containing tokenized fake news
    :param df_true: Pandas dataframe containing tokenized real news
    :param test_size: What fraction of the dataframe will be used for testing
    :return: df_train, df_test tuples
    """
    # Assign a label to each dataframe's texts
    df_fake['label'] = 0
    df_true['label'] = 1

    # Combine dataframes into a single one and drop useless columns after combining text + title
    df = pd.concat([df_fake, df_true], ignore_index=True)
    df['text'] = df['tokenized_text'] + df['tokenized_title']
    df = df.drop(columns=['tokenized_title', 'tokenized_text', 'date', 'subject', 'title'])
    
    # Shuffle dataframe and split into 80-20 train test.
    df = shuffle(df, random_state=42)
    df_train, df_test = train_test_split(df, test_size=test_size, shuffle=False)

    return df_train, df_test


def d2v_transform(corpus: List[List[str]], model: Doc2Vec) -> list:
    """
    Transform a list of texts to its vector representation
    :param corpus: corpus to transform (list of texts)
    :param model: Trained word2vec model
    :return: The vector form of the corpus
    """
    corpus_d2v = []
    for sample in corpus:
        if len(sample) > 0:
            # Infer the vector and append to the vectorized corpus
            corpus_d2v.append(model.infer_vector(sample))

    return corpus_d2v


def vectorize_dataframes(df_train: pd.DataFrame, df_test: pd.DataFrame, vectorizer: str, n_words: int = 5000) -> Tuple[
    List[List[str]], List[List[str]]]:
    """
    Vectorize the train/test corpus with sklearn or gensim's doc2vec
    :param df_train: Train dataframe
    :param df_test: Test dataframe
    :param vectorizer: sklearn's TfIdf or CountVectorizer, or gensim Doc2Vec
    :param n_words: How many words to keep from the corpora (top most frequent ones)
    :return: The vectorized corpora
    """
    _vectorizer = None
    X_train = df_train['text'].values.tolist()
    X_test = df_test['text'].values.tolist()

    if vectorizer in {'bow', 'tfidf'}:
        if vectorizer == 'bow':
            _vectorizer = CountVectorizer(tokenizer=lambda x: x, max_features=n_words, analyzer=lambda x: x)
        elif vectorizer == 'tfidf':
            _vectorizer = TfidfVectorizer(tokenizer=lambda x: x, max_features=n_words, analyzer=lambda x: x)

        X_train = _vectorizer.fit_transform(X_train)
        X_test = _vectorizer.transform(X_test)
    elif vectorizer == 'doc2vec':
        train_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(X_train)]
        d2v_model = Doc2Vec(vector_size=100, window=5, epochs=10, workers=6, min_count=5)
        d2v_model.build_vocab(train_documents)
        d2v_model.train(train_documents, total_examples=len(train_documents), epochs=d2v_model.epochs)

        X_train = d2v_transform(X_train, d2v_model)
        X_test = d2v_transform(X_test, d2v_model)

    return X_train, X_test
