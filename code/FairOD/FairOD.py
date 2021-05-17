"""
Implementation of competitive method FairOD: https://arxiv.org/abs/2012.03063

Date: 1/2021
"""
import sys, os
import torch
import time
import random
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from collections import defaultdict, Counter
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from Retriever import fetch

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
    torch.cuda.manual_seed_all(seed)
    torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False


class AE(nn.Module):
    """
    AutoEncoder
    """

    def __init__(self, d):
        super(AE, self).__init__()
        self.d = d
        nodes = 2
        if d > 100:
            nodes = 8
        self.ae = nn.Sequential(
            nn.Linear(d, nodes),
            nn.Linear(nodes, nodes),
            nn.Linear(nodes, d)
        ).to(cuda)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight).to(cuda)

    def forward(self, x):
        x = x.view(-1, self.d)
        x1 = self.ae(x)
        return x1.view(-1, 1, self.d)


def Train(model, dset, alpha, gamma, train_input, attribute, majority, minority, epochs, batch, base_score, one_hot, IDCG, PV_assignment, fair = False):
    """
    Model Training
    Args:
        model: model to be trained
        dset: dataset
        alpha: hyperparameter1
        gamma: hyperparameter2
        train_input: sample
        attribute: sensitive attribute subgroups
        majority: the majority subgroup
        minority: the minority subgroup
        epochs: training epoches
        batch: minibatch size
        base_score: the preobtained outlier score from the AE
        one_hot: one-hot-encoding of sensitive attribute subgroups
        IDCG: IDCG for each protected subgroup
        PV_assignment: The dict indicating each subgroup's index information, i.e., 1th, 3th, 5th belonging to male, 2th, 4th, 6th belonging to female
        fair: if False, will only train a standard AE, else will train FairOD.
                Note that FairOD require the output of AE

    Returns: statistical parity, group fidelity, and outlier score

    """
    model.train()
    optimizer = optim.Adam(model.parameters())
    running_loss = 0
    for epoch in range(epochs):
        for e in range(train_input.shape[0] // batch):
            cur_index = list(range(e * batch, (e + 1) * batch))
            input_batch = train_input[cur_index]
            x = torch.tensor(input_batch).float()
            x = x.to(cuda)

            x1 = model(x)
            if x.shape != x1.shape:
                x = np.reshape(x.data.cpu().numpy(), x1.shape)
                x = Variable(torch.tensor(x)).to(cuda)
            self_reconstruction_loss = nn.functional.mse_loss(x1, x, reduction='none').to(cuda)
            self_reconstruction_loss = torch.sum(self_reconstruction_loss, dim=2).to(cuda)
            self_reconstruction_loss = torch.reshape(self_reconstruction_loss, (self_reconstruction_loss.shape[0],))

            if fair:
                """
                in the paper: 
                one could one-hot-encode (OHE) the 𝑃𝑉 into multiple variables and minimize the correlation of 
                outlier scores with each variable additively. That is, an outer sum would be added to 
                Eq. (12) that goes over the new OHE variables encoding the categorical 𝑃𝑉 .
                """
                os_mean = torch.mean(self_reconstruction_loss)
                os_std = torch.std(self_reconstruction_loss)
                one_hot = torch.tensor(one_hot).to(cuda)
                Statistical_Parity = 0

                for i in range(one_hot.shape[1]):
                    cur_group = one_hot[:, i]
                    pv_mean = torch.mean(cur_group)
                    pv_std = torch.std(cur_group)
                    Statistical_Parity += abs(
                        (torch.sum(self_reconstruction_loss) - os_mean) * (torch.sum(cur_group) - pv_mean) / (os_std * pv_std))

                Group_Fidelity = len(set(attribute))
                for i in set(attribute):
                    group_batch_index = list(set().intersection(PV_assignment[i], cur_index))
                    for k in group_batch_index:
                        j = k - e * batch
                        Group_Fidelity -= (np.power(2, base_score[k]) - 1) / (IDCG[i] * torch.log2(1 +
                                                                torch.sum(torch.sigmoid(self_reconstruction_loss[group_batch_index] - self_reconstruction_loss[j]))))

                L = alpha * self_reconstruction_loss.mean() + (1 - alpha) * Statistical_Parity + gamma * Group_Fidelity
            else:
                L = self_reconstruction_loss.mean()
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            running_loss += L.data.cpu().numpy()
            if e % 10 == 9:  # show loss every 30 mini-batches
                print(f'[{epoch + 1},     {e + 1}] loss:{running_loss / 10:.2f}')
                running_loss = 0.0
    total = torch.tensor(train_input).float()
    total = total.to(cuda)
    final_x1 = model(total)
    if final_x1.shape != total.shape:
        total = np.reshape(total.data.cpu().numpy(), final_x1.shape)
        total = Variable(torch.tensor(total)).to(cuda)
    final_score = nn.functional.mse_loss(final_x1, total, reduction='none').to(cuda)
    final_score = torch.sum(final_score, dim=2).to(cuda)
    final_score = torch.reshape(final_score, (final_score.shape[0],))
    if not fair:
        np.save(f'baseline/{dset}.npy', final_score.data.cpu().numpy())
        return final_score.data.cpu().numpy()
    else:
        percentage = 80
        final_score_numpy = final_score.data.cpu().numpy()
        threshold = np.percentile(final_score_numpy, q=percentage)
        a = 0
        b = 0
        for i in range(train_input.shape[0]):
            if final_score_numpy[i] > threshold:
                if attribute[i] == majority[0]:
                    a += 1
                elif attribute[i] == minority[0]:
                    b += 1
        r = (a / majority[1]) / (b / minority[1])
        fairness = min(r, 1/r)

        NDCG_a = 0
        NDCG_b = 0
        for j in PV_assignment[majority[0]]:
            sum = 1 + torch.sum(final_score[PV_assignment[majority[0]]] >= final_score[j])
            NDCG_a += (torch.pow(2, torch.tensor(base_score[j]).to(cuda)) - 1) / (IDCG[majority[0]] * torch.log2(sum.float()))

        for j in PV_assignment[minority[0]]:
            sum = 1 + torch.sum(final_score[PV_assignment[minority[0]]] >= final_score[j])
            NDCG_b += (torch.pow(2, torch.tensor(base_score[j]).to(cuda)) - 1) / (IDCG[minority[0]] * torch.log2(sum.float()))
        group_fidelity = 1 / (1 / NDCG_a.data.cpu().numpy() + 1/NDCG_b.data.cpu().numpy())
        print('fairness:', fairness)
        print('group_fidelity:', group_fidelity)
        return fairness, group_fidelity, final_score


def main():
    db = sys.argv[1]
    X_norm = np.load(f'../../datasets/{db.upper()}_X.npy')
    Y = np.load(f'../../datasets/{db.upper()}_Y.npy')
    sensitive_attribute_group = np.load(f'../{db}/attribute.npy', allow_pickle=True)
    input = np.reshape(sensitive_attribute_group, (-1, 1))
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(input)
    one_hot = enc.transform(input).toarray()
    sensitive_attribute_group = np.argmax(one_hot, axis=1)

    # -----data shuffling----- #
    set_seed(2021)
    random_index = np.random.permutation(X_norm.shape[0])
    X_norm = X_norm[random_index]
    Y = Y[random_index]
    sensitive_attribute_group = sensitive_attribute_group[random_index]

    # -----setting epoch and minibatch size----- #
    starttime = time.time()
    if X_norm.shape[0] < 10000:
        epoch, batch = 90, 64
    else:
        epoch, batch = 10, 512

    fair_command = sys.argv[3]
    # -----train AE----- #
    if not os.path.exists('baseline'):
        os.makedirs('baseline')
    if fair_command == 'f':
        ae = AE(X_norm.shape[1])
        score = Train(ae, db, 0, 0, X_norm, [], [], [], epoch, batch, NotImplemented, NotImplemented, NotImplemented, NotImplemented, fair=False)
        print('base AUC:', roc_auc_score(Y, score))
        fgap, frank, _ = fetch(score,
                                 f'../../datasets/{db.upper()}_Y.npy',
                                 f'../{db}/attribute.npy')
        return
    else:
        fair = True

    # -----cache IDCG and PV_assignment to facilitate model training----- #
    base_score = np.load(f'baseline/{db}.npy')
    base_score = (base_score - min(base_score)) / (max(base_score) - min(base_score))
    IDCG = defaultdict(float)
    PV_assignment = defaultdict(list)
    for i in range(X_norm.shape[0]):
        PV_assignment[sensitive_attribute_group[i]].append(i)

    for i in set(sensitive_attribute_group):
        scores = torch.tensor(base_score[PV_assignment[i]]).to(cuda)
        scores, _ = torch.sort(scores, descending=True)
        scores = scores.data.cpu().numpy()
        for j in range(len(scores)):
            IDCG[i] += (np.power(2, scores[j]) - 1) / np.log2(1 + (j + 1))

    c = Counter(sensitive_attribute_group)
    majority = c.most_common()[0]
    minority = c.most_common()[-1]

    auc=[]
    gap= []
    rank= []
    random_run = 5
    for e in range(random_run):
        set_seed(e)
        alpha = [0.01, 0.5, 0.9]
        gamma = [0.01, 0.1, 1.0]
        best_choice = NotImplemented
        for i in alpha:
            for j in gamma:
                print(f'Using alpha: {i},   gamma:{j}')
                ae = AE(X_norm.shape[1])
                if db == 'kdd':
                    ae = nn.DataParallel(ae, device_ids=devices).to(cuda)
                fairness, group_fidelity, final_scores = \
                    Train(ae, db, i, j, X_norm, sensitive_attribute_group, majority, minority, epoch, batch, base_score, one_hot, IDCG, PV_assignment, fair=fair)
                # -----find the best combination of alpha and gamma in the hyperparameter grid ----- #
                if best_choice is NotImplemented:
                    best_choice = (fairness, group_fidelity, final_scores)
                else:
                    current_best = best_choice[0] + best_choice[1]
                    if current_best < fairness + group_fidelity:
                        best_choice = (fairness, group_fidelity, final_scores)
        auc_score = roc_auc_score(Y, best_choice[2].data.cpu().numpy())
        print(f'auc: {auc_score}')
        fgap, frank, _ = fetch(best_choice[2],
                               f'../../datasets/{db.upper()}_Y.npy',
                               f'../{db}/attribute.npy')
        auc.append(auc_score)
        gap.append(fgap)
        rank.append(frank)
        if e == 0:
            time_taken = time.time() - starttime
            print(f'time for {db}:', time_taken)
    print('AUC:', auc)
    print('Fgap:', gap)
    print('Frank:', rank)
    print(f'The mean AUC is: {np.mean(auc)}, the std is {np.std(auc)}\n')
    print(f'The mean Fgap is: {np.mean(gap)}, the std is {np.std(gap)}\n')
    print(f'The mean Frank is: {np.mean(rank)}, the std is {np.std(rank)}\n')


if __name__ == '__main__':
    main()