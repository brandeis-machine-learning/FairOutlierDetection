"""
Script to get all the fairness metrics results

Date: 11/2020
"""
import sys
import os
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import roc_auc_score


def kl_divergence(p, q):
    return sum(p[i] * np.log(p[i] / q[i]) for i in range(len(p)))


def normalize(score):
    span = max(score) - min(score)
    min_element = min(score)
    normalized = []
    for i in score:
        normalized.append((i - min_element) / span)
    return np.array(normalized)


def fetch(temp_score, label_path, group_path):
    """
    fetch the fairness metrics Fgap and Frank for a dataset with the obtained score

    Args:
        temp_score: outlier score
        label_path: path to the true outlier labels
        group_path: path to the sensitive attribute subgroup information

    Returns: Fgap, Frank, subgroup distribution in Frank

    """
    sensitive_attribute_group = np.load(group_path, allow_pickle=True)
    labels = np.load(label_path)
    temp_score = normalize(temp_score)
    res1 = gap(temp_score, labels, sensitive_attribute_group)
    res2, res3 = rank(20, temp_score, sensitive_attribute_group)

    print(f'Fgap: {round(res1, ndigits=4)}, '
            f'Frank: {round(res2, ndigits=4)}')
    return res1, res2, res3


def gap(temp_score, true_labels, attribute):
    """
    Fgap: the outlier detection validity gap between sensitive subgroups
    Args:
        temp_score: outlier score
        true_labels: true outlier labels
        attribute: sensitive attribute subgroups

    Returns: Fgap

    """
    length = len(temp_score)
    attribute_indices = defaultdict(list)
    for i in range(length):
        attribute_indices[attribute[i]].append(i)

    unique_attri = set(attribute)
    res = []
    for i in unique_attri:
        indices = attribute_indices[i]
        current_score = temp_score[indices]
        current_truth = true_labels[indices]
        res.append(roc_auc_score(current_truth, current_score))

    return max(res) - min(res)


def rank(percentage, temp_score, attribute):
    """
    Frank: the most significant distribution drift in top 5% - 20% outlier candidates,
            compared to the reference distribution

    Args:
        percentage: top % outlier candidates
        temp_score: outlier score
        attribute: sensitive attribute subgroups

    Returns: Frank

    """
    total_size = len(temp_score)
    unique_attri = set(attribute)
    optimal_distribution = []
    attri_count = Counter()

    for i in attribute:
        attri_count[i] += 1

    for i in sorted(unique_attri):
        optimal_distribution.append(attri_count[i] / total_size)

    highest_distribution = NotImplemented
    highest_kld = 0.0
    for i in range(percentage - 4):
        temp_distribution = []
        partition_size = int((i + 5) / 100 * total_size)
        indices = np.argsort(-temp_score)[:partition_size]
        attribute_count = Counter()

        for j in attribute[indices]:
            attribute_count[j] += 1

        for j in sorted(attribute_count.keys()):
            temp_distribution.append(attribute_count[j] / len(indices))
        temp_distribution = np.array(temp_distribution)
        current_kld = kl_divergence(temp_distribution, optimal_distribution)
        if current_kld > highest_kld:
            highest_kld = current_kld
            highest_distribution = temp_distribution

    print('subgroup distribution:', highest_distribution)
    return highest_kld, highest_distribution


def main():
    db = sys.argv[1]
    path = f'{db}/'
    all_methods = []
    all_methods += [each for each in os.listdir(path) if each.endswith('raw.npy')]
    sensitive_attribute_group = np.load(f'{db}/attribute.npy', allow_pickle=True)
    labels = np.load(f'../datasets/{db.upper()}_Y.npy')

    for i in all_methods:
        temp_score = np.load(f'{db}/{i}')
        temp_score = normalize(temp_score)

        res1 = gap(temp_score, labels, sensitive_attribute_group)
        res2 = rank(20, temp_score, sensitive_attribute_group)[0]
        index = i.find('_')
        print((f'{i[:index]} '
               f'Fgap: {round(res1, ndigits=4)}, '
               f'Frank: {round(res2, ndigits=4)}\n'))


if __name__ == '__main__':
    main()
