import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def show_embeddings(vectors, labels, TSNE_args=None, legend_title=None, label_dict=None):
    """Визуализация эмбеддингов на двумерной плоскости
    """
    # reduce dimensions with TSNE
    if TSNE_args is None:
        TSNE_args = {'n_components': 2,
                     'perplexity': 40,
                     'random_state': 22,
                     'learning_rate': 'auto',
                     'init': 'random'}
    trans_vectors = TSNE(**TSNE_args).fit_transform(vectors)

    patches = []
    unique_labels = np.unique(labels)

    if legend_title is None:
        legend_title = labels.name

    if label_dict is None:
        label_dict = {k: k for k in unique_labels}

    for label in unique_labels:
        indexes = np.where(labels == label)[0]
        g = plt.scatter(
            trans_vectors[indexes, 0],
            trans_vectors[indexes, 1],
            alpha=0.2, linewidths=0,
        )
        label_color = g.get_facecolor()  # цвет лэйбла
        label_color[0, -1] = 1.  # делаем цвет непрозрачным
        patches.append(mpatches.Patch(color=label_color, label=label_dict[label]))
    plt.legend(handles=patches, title=legend_title)
    return
