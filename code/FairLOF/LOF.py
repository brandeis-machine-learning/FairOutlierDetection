"""
Implementation for paper: https://arxiv.org/abs/2005.09900

Date: 12/2020

"""
from __future__ import division
import sys
import time
import random
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import defaultdict

cuda0 = torch.device('cuda:0')
cuda1 = torch.device('cuda:1')
cuda2 = torch.device('cuda:2')
cuda3 = torch.device('cuda:3')


def set_seed(seed):
    print(f"setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False


def distance_euclidean(middle, instance_list):
    """
    Split the data into two parts and have four matrices on four GPUs to save memory
    Args:
        middle: the middle index
        instance_list: the list needed to be split

    Returns: 4 matrices, altogether create a distance matrix between each instance in the original instance list

    """
    first_half = instance_list[:middle]
    second_half = instance_list[middle:]
    matrix11 = torch.cdist(torch.tensor(first_half).to(cuda0), torch.tensor(first_half).to(cuda0))
    matrix12 = torch.cdist(torch.tensor(first_half).to(cuda1), torch.tensor(second_half).to(cuda1))
    matrix21 = torch.cdist(torch.tensor(second_half).to(cuda2), torch.tensor(first_half).to(cuda2))
    matrix22 = torch.cdist(torch.tensor(second_half).to(cuda3), torch.tensor(second_half).to(cuda3))
    return matrix11, matrix12, matrix21, matrix22


class LOF:
    def __init__(self, k, instances, distance_function=distance_euclidean):
        self.k = k
        self.middle = 0
        self.instances = instances
        self.distance_function = distance_function
        self.kdistance_values = defaultdict(float)
        self.kneighbors = defaultdict(list)
        self.distance_matrices = NotImplemented
        self.density_map = defaultdict(float)
        self.set_up()

    def set_up(self):
        """ Caching neighbors for all instances """
        print('Setting up...')
        middle = int(len(self.instances) / 2)
        self.middle = middle
        self.distance_matrices = self.distance_function(middle, self.instances)

        for i in range(len(self.instances)):
            if i < middle:
                if i < middle / 2:
                    temp_cuda = cuda0
                else:
                    temp_cuda = cuda1
                distance_list = torch.cat(
                    (self.distance_matrices[0][i, :].to(temp_cuda), self.distance_matrices[1][i, :].to(temp_cuda)), -1)
            else:
                if i < middle * 1.5:
                    temp_cuda = cuda2
                else:
                    temp_cuda = cuda3
                distance_list = torch.cat((self.distance_matrices[2][i - middle, :].to(temp_cuda),
                                           self.distance_matrices[3][i - middle, :].to(temp_cuda)), -1)
            all_distances, all_distances_index = torch.sort(distance_list, descending=False)
            count = 0
            current_neighbors = []
            last_distance = NotImplemented

            for j in range(len(all_distances)):
                if j == 0:
                    continue
                cur_distance = all_distances[j]
                if last_distance is NotImplemented or cur_distance != last_distance:
                    count += 1
                last_distance = cur_distance
                current_neighbors.append(int(all_distances_index[j]))
                if count == self.k:
                    self.kdistance_values[i] = last_distance
                    self.kneighbors[i] = current_neighbors
                    break
        print('Finished setting up!')

    def get_value_neighbors(self, index):
        return self.kdistance_values[index], self.kneighbors[index]

    def local_outlier_factor(self, index):
        (k_distance_value, neighbours) = self.get_value_neighbors(index)
        if self.density_map[index] == 0.0:
            instance_lrd = self.local_reachability_density(index)
            self.density_map[index] = instance_lrd
        else:
            instance_lrd = self.density_map[index]

        lrd_ratios_array = [0] * len(neighbours)
        for i, neighbour_index in enumerate(neighbours):
            if self.density_map[neighbour_index] == 0.0:
                neighbour_lrd = self.local_reachability_density(neighbour_index)
                self.density_map[neighbour_index] = neighbour_lrd
            else:
                neighbour_lrd = self.density_map[neighbour_index]
            lrd_ratios_array[i] = neighbour_lrd / instance_lrd
        return sum(lrd_ratios_array) / len(neighbours)

    def local_reachability_density(self, index):
        (k_distance_value, neighbours) = self.get_value_neighbors(index)
        reachability_distances_array = [0] * len(neighbours)
        for i, neighbour in enumerate(neighbours):
            temp_index = index
            temp_neighbor = neighbour
            if index < self.middle:
                if neighbour < self.middle:
                    temp_cuda = cuda0
                    case = 0
                else:
                    temp_cuda = cuda1
                    case = 1
                    temp_neighbor -= self.middle
            else:
                temp_index -= self.middle
                if neighbour < self.middle:
                    temp_cuda = cuda2
                    case = 2
                else:
                    temp_cuda = cuda3
                    case = 3
                    temp_neighbor -= self.middle

            reachability_distances_array[i] = float(torch.max(self.kdistance_values[neighbour].to(temp_cuda),
                                                        self.distance_matrices[case][temp_index][temp_neighbor].to(temp_cuda)))
        return len(neighbours) / sum(reachability_distances_array)


def outlierScore(db, k, instances, **kwargs):
    scores = []
    l = LOF(k, instances, **kwargs)
    for i, cur in enumerate(instances):
        score = l.local_outlier_factor(i)
        scores.append(float(score))
    scores = np.array(scores)
    np.save(f'../{db}/mylof.npy', scores)
    return scores


def main():
    set_seed(1)
    db = sys.argv[1]
    X_arrays = np.load(f'../../datasets/{db.upper()}_X.npy')
    X_norm = []
    for i in X_arrays:
        X_norm.append(i)
    Y_arrays = np.load(f'../../datasets/{db.upper()}_Y.npy')
    Y = []
    for i in Y_arrays:
        Y.append(i)
    starttime = time.time()
    scores = outlierScore(db, 5, X_norm)
    print(f'Time used: {time.time() - starttime} seconds')
    print(f'AUC for {db} is:', roc_auc_score(Y, scores))


if __name__ == '__main__':
    main()
