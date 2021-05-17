"""
Obtain the results for methods implemented in Pyod
"""
import sys
import time
import numpy as np

from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM
from pyod.models.lof import LOF
from pyod.models.cof import COF
from pyod.models.cblof import CBLOF
from pyod.models.abod import ABOD
from pyod.models.copod import COPOD
from pyod.models.iforest import IForest
from pyod.models.loda import LODA
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE

from sklearn.metrics import roc_auc_score
from Retriever import fetch, normalize


def compare(inputdata, labels, n_clusters, dset_name):
    """
    Compute the AUC, Fgap, Frank score on all conventional outlier detectors for the given dataset
    Args:
        inputdata: input data
        labels: ground truth outlier labels
        n_clusters: number of clusters, for some cluster-based detectors
        dset_name: dataset

    Returns: AUC, Fgap, Frank

    """
    print("Competing with conventional unsupervised outlier detection algorithms...")
    random_state = np.random.RandomState(1)
    if inputdata.shape[1] < 64:
        AEneurons = [16, 8, 8, 16]
        VAEneurons = [16, 8, 4], [4, 8, 16]
    else:
        AEneurons = [64, 32, 32, 64]
        VAEneurons = [128, 64, 32], [32, 64, 128]

    classifiers = {
        'PCA' : PCA(random_state=random_state),
        'AutoEncoder': AutoEncoder(batch_size=100, hidden_neurons= AEneurons, random_state=random_state),
        'VAE': VAE(batch_size=100, encoder_neurons=VAEneurons[0],decoder_neurons=VAEneurons[1], random_state=random_state),
        'COPOD' : COPOD(),
        'Iforest' : IForest(random_state=random_state),
        'AutoEncoder': AutoEncoder(batch_size=100, random_state=random_state),
        'VAE': VAE(batch_size=100, random_state=random_state),
        'LODA' : LODA(),
        'OCSVM': OCSVM(),
        'ABOD': ABOD(n_neighbors=20),
        'Fb': FeatureBagging(random_state=random_state),
        'CBLOF': CBLOF(
            n_clusters=n_clusters,
            check_estimator=False,
            random_state=random_state),
        'LOF': LOF(),
        'COF': COF()
    }

    for clf_name, clf in classifiers.items():
        print(f"Using {clf_name} method")
        starttime = time.time()
        clf.fit(inputdata)
        time_taken = time.time() - starttime
        test_scores = clf.decision_scores_

        # -----fix some broken scores----- #
        for i in range(len(test_scores)):
            cur = test_scores[i]
            if np.isnan(cur) or not np.isfinite(cur):
                test_scores[i] = 0

        np.save(f'{dset_name}/{clf_name}_raw.npy', test_scores)
        auc = roc_auc_score(labels, test_scores)
        print('AUC:', auc)
        fetch(normalize(test_scores),
                                         f'../datasets/{dset_name.upper()}_Y.npy',
                                         f'{dset_name}/attribute.npy')
        print('time_taken:', time_taken)


def main():
    db = sys.argv[1]
    num_centroid = 10
    X_norm = np.load(f'../datasets/{db.upper()}_X.npy')
    Y = np.load(f'../datasets/{db.upper()}_Y.npy')
    compare(X_norm, Y, num_centroid, db)


if __name__ == '__main__':
    main()