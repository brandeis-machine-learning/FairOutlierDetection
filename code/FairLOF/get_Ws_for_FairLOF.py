"""
Retrieves the necessary metrics for FairLOF with the results of LOF
"""
import numpy as np
import torch
from collections import Counter
from sklearn.preprocessing import OneHotEncoder

cuda = torch.device('cuda:0')


def main():
    datasets = ['cc', 'adult', 'student', 'drug', 'crime', 'german', 'asd', 'obesity']
    for db in datasets:
        scores = np.load(f'../{db}/mylof.npy', allow_pickle=True)
        t = min(500, int(0.05 * len(scores)))
        _, indices = torch.sort(torch.tensor(scores).to(cuda))
        top_t_indices = indices[:t].tolist()

        sensitive_attribute_group = np.load(f'../{db}/attribute.npy', allow_pickle=True)
        input = np.reshape(sensitive_attribute_group, (-1, 1))
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(input)
        one_hot = enc.transform(input).toarray()
        sensitive_attribute_group = np.argmax(one_hot, axis=1)
        top_t_sag = sensitive_attribute_group[top_t_indices]

        counter = Counter()
        for i in set(sensitive_attribute_group):
            counter[i] = 0
        for i in top_t_sag:
            counter[i] += 1
        res = []
        for i in sorted(counter.keys()):
            res.append(counter[i] / t)
        print(f'{db} distribution is: {res}')
        np.save(f'../{db}/Ws.npy', np.array(res))


if __name__ == '__main__':
    main()