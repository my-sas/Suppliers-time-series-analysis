import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def show_embeddings(vectors, labels=None):
    # reduce dimensions with TSNE
    trans_vectors = TSNE(
        n_components=2, learning_rate='auto', init='random',
        perplexity=40, random_state=22
    ).fit_transform(vectors)

    for label in np.unique(labels):
        indexes = np.where(labels == label)[0]
        plt.scatter(
            trans_vectors[indexes, 0],
            trans_vectors[indexes, 1],
            alpha=0.2, linewidths=0,
        )
    return
