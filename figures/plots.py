import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def show_embeddings(vectors, labels):
    """Визуализация эмбеддингов на двумерной плоскости
    """

    # reduce dimensions with TSNE
    trans_vectors = TSNE(
        n_components=2, learning_rate='auto', init='random',
        perplexity=40, random_state=22
    ).fit_transform(vectors)

    patches = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        indexes = np.where(labels == label)[0]
        g = plt.scatter(
            trans_vectors[indexes, 0],
            trans_vectors[indexes, 1],
            alpha=0.2, linewidths=0,
        )
        label_color = g.get_facecolor()  # цвет лэйбла
        label_color[0, -1] = 1.  # делаем цвет непрозрачным
        patches.append(mpatches.Patch(color=label_color, label=label))
    plt.legend(handles=patches, title=labels.name)
    return
