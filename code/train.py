"""
Code for Deep Clustering Fair Outlier Detection

Date: 11/2020
"""

import sys
import torch
import time
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from Retriever import fetch
from model import DCFOD
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# -----indicate which gpu to use for training, devices list will be used in training with DataParellel----- #
gpu = sys.argv[2]
if gpu == '0':
    cuda = torch.device('cuda:0')
    devices = [0, 1, 2, 3]
elif gpu == '1':
    cuda = torch.device('cuda:1')
    devices = [1, 2, 3, 0]
elif gpu == '2':
    cuda = torch.device('cuda:2')
    devices = [2, 3, 0, 1]
elif gpu == '3':
    cuda = torch.device('cuda:3')
    devices = [3, 0, 1, 2]
else:
    raise NameError('no more GPUs')


def set_seed(seed):
    print(f"setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False


def acc(dset, Y, dist):
    """
    Calculate the AUC, Fgap, and Frank
    Args:
        dset: dataset
        Y: ground truth outlier label
        dist: distance to cluster centers

    Returns: AUC, Fgap, Frank

    """

    outlier_score, position = torch.min(dist, dim=1)
    for i in range(dist.shape[1]):
        pos = list(x for x in range(len(outlier_score)) if position[x] == i)
        if len(outlier_score[pos]) != 0:
            max_dist = max(outlier_score[pos])
            outlier_score[pos] = torch.div(outlier_score[pos], max_dist).to(cuda)
    Fgap, Frank, _ = fetch(outlier_score.data.cpu(),
                           f'../datasets/{dset.upper()}_Y.npy',
                           f'{dset}/attribute.npy')
    AUC = roc_auc_score(Y, outlier_score.data.cpu().numpy())
    return AUC, Fgap, Frank


def target_distribution(q):
    """
    Calculate the auxiliary distribution with the original distribution
    Args:
        q: original distribution

    Returns: auxiliary distribution

    """
    weight = (q ** 2) / q.sum(0)
    return torch.div(weight.t(), weight.sum(1)).t().data


def kld(q, p):
    """
    KL-divergence
    Args:
        q: original distribution
        p: auxiliary distribution

    Returns: the similarity between two probability distributions

    """
    return torch.sum(p * torch.log(p / q).to(cuda), dim=-1)


def getTDistribution(model, x):
    """
    Obtain the distance to centroid for each instance, and calculate the weight module based on that
    Args:
        model: DCFOD
        x: embedded x

    Returns: weight module, clustering distribution

    """

    # dist, dist_to_centers = model.module.getDistanceToClusters(x)
    dist, dist_to_centers = model.getDistanceToClusters(x)

    # -----find the centroid for each instance, with their distance in between----- #
    outlier_score, centroid = torch.min(dist_to_centers, dim=1)

    # -----for each instance, calculate a score
    # by the outlier_score divided by the furtherest instance in the centroid----- #
    for i in range(dist_to_centers.shape[1]):
        pos = list(x for x in range(len(outlier_score)) if centroid[x] == i)
        if len(outlier_score[pos]) != 0:
            max_dist = max(outlier_score[pos])
            outlier_score[pos] = torch.div(outlier_score[pos], max_dist).to(cuda)
    sm = nn.Softmax(dim=0).to(cuda)
    weight = sm(outlier_score.neg())

    # -----calculate the clustering distribution with the distance----- #
    # q = 1.0 / (1.0 + (dist / model.module.alpha))
    q = 1.0 / (1.0 + (dist / model.alpha))
    # q = q ** (model.module.alpha + 1.0) / 2.0
    q = q ** (model.alpha + 1.0) / 2.0
    q = (q.t() / torch.sum(q, 1)).t()
    return weight, q


def clustering(model, mbk, x):
    """
    Initialize cluster centroids with minibatch Kmeans
    Args:
        model: DCFOD
        mbk: minibatch Kmeans
        x: embedded x

    Returns: N/A

    """
    model.eval()
    x_e = model(x.float())
    mbk.partial_fit(x_e.data.cpu().numpy())
    # model.module.cluster_centers = mbk.cluster_centers_  # keep the cluster centers
    # model.module.clusterCenter.data = torch.from_numpy(model.module.cluster_centers).to(cuda)
    model.cluster_centers = mbk.cluster_centers_  # keep the cluster centers
    model.clusterCenter.data = torch.from_numpy(model.cluster_centers).to(cuda)


def Train(model, dset, train_input, labels, attribute, epochs, batch, with_weight=False, ks=8, kf=100):
    """
    Train DCFOD in minibatch
    Args:
        model: DCFOD
        dset: dataset
        train_input: input data
        labels: ground truth outlier score, which will not be used during training
        attribute: sensitive attribute subgroups
        epochs: total number of iterations
        batch: minibatch size
        with_weight: if training with weight
        ks: hyperparameter for self-reconstruction loss
        kf: hyperparameter for fairness-adversarial loss

    Returns: AUC, Fgap, Frank

    """
    model.train()
    # if dset == 'kdd':
    #     model = model.module
    # mbk = MiniBatchKMeans(n_clusters=model.module.num_classes, n_init=20, batch_size=batch)
    mbk = MiniBatchKMeans(n_clusters=model.num_classes, n_init=20, batch_size=batch)
    got_cluster_center = False
    running_loss = 0.0
    fair_loss = 0.0
    lr_cluster = 0.0001
    lr_discriminator = 0.00001
    lr_sae = 0.00001

    # optimizer = optim.SGD([
    #     {'params': model.module.encoder.parameters()},
    #     {'params': model.module.decoder.parameters()},
    #     {'params': model.module.discriminator.parameters(), 'lr': lr_discriminator},
    #     {'params': model.module.clusterCenter, 'lr': lr_cluster}
    # ], lr=lr_sae, momentum=0.9)
    optimizer = optim.SGD([
        {'params': model.encoder.parameters()},
        {'params': model.decoder.parameters()},
        {'params': model.discriminator.parameters(), 'lr': lr_discriminator},
        {'params': model.clusterCenter, 'lr': lr_cluster}
    ], lr=lr_sae, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print(f'Learning rate: {lr_cluster}, {lr_sae}, {lr_discriminator}')
    print(f'batch size: {batch}, self_recon: {ks}, fairness: {kf}')

    for epoch in range(epochs):
        for i in range(train_input.shape[0] // batch):
            input_batch = train_input[i * batch: (i + 1) * batch]
            x = torch.tensor(input_batch).float()
            x = x.to(cuda)

            attribute_batch = attribute[i * batch: (i + 1) * batch]
            attribute_batch = torch.tensor(attribute_batch).to(cuda)

            # -----use minibatch Kmeans to initialize the cluster centroids for the clustering layer----- #
            if not got_cluster_center:
                # model.module.set_clustering_mode(True)
                model.setClusteringMode(True)
                total = torch.tensor(train_input).to(cuda)
                clustering(model, mbk, total)
                got_cluster_center = True
                # model.module.set_clustering_mode(False)
                model.setClusteringMode(False)
            else:
                model.train()
                x_e, x_de, x_sa = model(x)

                # -----obtain the clustering probability distribution and dynamic weight----- #
                weight, q = getTDistribution(model, x_e)
                if x.shape != x_de.shape:
                    x = np.reshape(x.data.cpu().numpy(), x_de.shape)
                    x = torch.tensor(x).to(cuda)
                p = target_distribution(q)
                clustering_regularizer_loss = kld(q, p)

                self_reconstruction_loss = nn.functional.mse_loss(x_de, x, reduction='none').to(cuda)
                self_reconstruction_loss = torch.sum(self_reconstruction_loss, dim=2)
                self_reconstruction_loss = torch.reshape(self_reconstruction_loss, (self_reconstruction_loss.shape[0],))

                CELoss = nn.CrossEntropyLoss().to(cuda)
                discriminator_loss = CELoss(x_sa, attribute_batch)

                objective = ks * self_reconstruction_loss + kf * discriminator_loss + clustering_regularizer_loss
                if with_weight:
                    L = torch.sum(torch.mul(objective, weight))
                else:
                    L = objective.mean()
                optimizer.zero_grad()
                L.backward()
                optimizer.step()
                running_loss += L.data.cpu().numpy()
                fair_loss += discriminator_loss.data.cpu().numpy()

                # -----show loss every 20 mini-batches----- #
                if i % 30 == 29:
                    print(f'[{epoch + 1},     {i + 1}] L:{running_loss / 30:.2f}, FairLoss: {fair_loss / 30:.4f}')
                    running_loss = 0.0
                    fair_loss = 0.0

        # -----after one training iteration, validate the performance on the whole dataset----- #
        if dset != 'kdd':
            res = validate(model, dset, train_input, labels)
        else:
            res = validate(model, dset, train_input, labels)
        scheduler.step()

    print('Done Training.')
    print(f'AUC: {res[0]}')
    print(f'Fgap: {res[1]}')
    print(f'Frank: {res[2]}')

    return res[0].data.cpu(), res[1], res[2]


def validate(model, dset, train_input, Y):
    """
    check the model performance after one iteration of minibatch training
    Args:
        model: DCFOD
        dset: dataset
        train_input: input data
        Y: ground truth outlier labels

    Returns: AUC, Fgap, Frank

    """

    # -----empty cache to save memory for kdd dataset, or have to use DataParellel----- #
    torch.cuda.empty_cache()
    model.eval()

    # -----set model to validate mode, so it only returns the embedded space----- #
    # model.module.setTrainValidateMode(True)
    model.setValidateMode(True)
    model_input = torch.tensor(train_input).to(cuda)
    xe = model(model_input.float())

    # -----obtain all instances' distance to cluster centroids----- #
    # _, dist = model.module.getDistanceToClusters(x)
    _, dist = model.getDistanceToClusters(xe)

    # -----set to retrieve AUC, Fgap, Frank values in acc function----- #
    res = acc(dset, Y, dist)
    # model.module.setTrainValidateMode(False)
    model.setValidateMode(False)
    print(' ' * 8 + '|==>  AUC: %.4f <==|' % res[0])
    return res


def shuffle(X, Y, S):
    """
    Shuffle the datasets
    Args:
        X: input data
        Y: outlier labels
        S: sensitive attribute subgroups

    Returns: shuffled sets

    """
    set_seed(2021)
    random_index = np.random.permutation(X.shape[0])
    return X[random_index], Y[random_index], S[random_index]


def main():
    db = sys.argv[1]
    with_weight = sys.argv[3]
    if with_weight == 'true':
        weight = True
    else:
        weight = False

    # -----load sensitive subgroups----- #
    sensitive_attribute_group = np.load(f'{db}/attribute.npy', allow_pickle=True)
    input = np.reshape(sensitive_attribute_group, (-1, 1))
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(input)
    one_hot = enc.transform(input).toarray()
    sensitive_attribute_group = np.argmax(one_hot, axis=1)

    # -----load dataset----- #
    X_norm = np.load(f'../datasets/{db.upper()}_X.npy')
    Y = np.load(f'../datasets/{db.upper()}_Y.npy')
    X_norm, Y, sensitive_attribute_group = shuffle(X_norm, Y, sensitive_attribute_group)

    num_centroid = 10
    feature_dimension = X_norm.shape[1]
    embedded_dimension = 64
    random_run = 20
    num_subgroups = len(set(sensitive_attribute_group))
    configuration = 90, 64 if X_norm.shape[0] < 10000 else 40, 256

    # -----run the model in 20 random seeds and report the average----- #
    AUC = []
    Fgap = []
    Frank = []
    Times = []
    for i in range(random_run):
        starttime = time.time()
        set_seed(i)
        model = DCFOD(feature_dimension, num_centroid, embedded_dimension, num_subgroups, cuda)

        # -----if the memory space on one gpu is not sufficient, use Dataparellel to run on multiple gpus----- #
        # model = nn.DataParallel(model, device_ids=devices).to(cuda)
        res = Train(model, db, X_norm, Y, sensitive_attribute_group,
                    configuration[0], configuration[1], with_weight=weight)
        Times.append(time.time() - starttime)
        AUC.append(float(res[0]))
        Fgap.append(res[1])
        Frank.append(res[2])
        print(f'End of Training for seed {i}')
    print('Average time:', round(sum(Times) / random_run))
    print(f'The mean AUC is: {np.mean(AUC)}, the std is {np.std(AUC)}\n')
    print(f'The mean Fgap is: {np.mean(Fgap)}, the std is {np.std(Fgap)}\n')
    print(f'The mean Frank is: {np.mean(Frank)}, the std is {np.std(Frank)}\n')


if __name__ == '__main__':
    main()
