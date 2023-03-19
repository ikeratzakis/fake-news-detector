import time
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def plot_wordclouds(df_fake: pd.DataFrame, df_true: pd.DataFrame):
    print("Generating wordclouds...")
    start_time = time.time()

    fake_cloud = WordCloud(background_color='white').generate("".join(title for title in df_fake['title']))
    true_cloud = WordCloud(background_color='white').generate("".join(title for title in df_true['title']))

    print(f"Computed wordclouds in {time.time() - start_time} seconds")
    figure, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].set_title('Fake news')
    axes[0].imshow(fake_cloud, interpolation='bilinear')
    axes[1].set_title('True news')
    axes[1].imshow(true_cloud, interpolation='bilinear')
    figure.tight_layout()
    plt.show()

    # Ram cleanup
    del fake_cloud
    del true_cloud


def plot_title_lengths(df_fake: pd.DataFrame, df_true: pd.DataFrame):
    figure, axes = plt.subplots(nrows=2, ncols=2)
    axes[0, 0].set_title('Fake title length distribution')
    axes[0, 0].hist([len(title.split()) for title in df_fake['title']])
    axes[0, 1].set_title('True title length distribution')
    axes[0, 1].hist([len(title.split()) for title in df_true['title']])
    axes[1, 0].set_title('Fake news length distribution')
    axes[1, 0].hist([len(text.split()) for text in df_fake['text']])
    axes[1, 1].set_title('True news length distribution')
    axes[1, 1].hist([len(text.split()) for text in df_true['text']])

    figure.tight_layout()
    plt.show()
