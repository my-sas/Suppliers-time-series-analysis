import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def show_embeddings(vectors, labels=None):
    # reduce dimensions with TSNE
    trans_vectors = TSNE(
        n_components=2, learning_rate='auto', init='random', perplexity=5
    ).fit_transform(vectors)

    # create legend if labels are known
    colors = None
    if labels is not None:
        # create color map from unique labels
        unique_labels = np.unique(labels)
        label_idx = {k: v for k, v in zip(unique_labels, range(len(unique_labels)))}
        cmap = plt.cm.get_cmap('gist_rainbow', len(unique_labels))
        colors = [cmap(label_idx[i]) for i in labels]

        # add legend with labels manually
        patches = [mpatches.Patch(color=cmap(i), label=label) for i, label in enumerate(unique_labels)]
        plt.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    plt.scatter(
        trans_vectors[:, 0],
        trans_vectors[:, 1],
        c=colors, alpha=1
    )
    return
