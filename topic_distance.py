import numpy as np
import pandas as pd
from conference import Talk
from collections import defaultdict

class TopicDistance:
    def __init__(self, fname_distances='topic_dist_2020_10_3.npy', fname_submissions='accepted_submissions_2020_10_3.csv'):
        self.D = D = np.load(fname_distances)
        self.talks = talks = pd.read_csv(fname_submissions)
        self.talk_to_idx = {}
        self.missing = set()
        # np.fill_diagonal(D, 0)
        # for _ in range(100):
        #     i, j = np.unravel_index(np.argmax(D), D.shape)
        #     print(f'{D[i, j]}:')
        #     print(f'  - {talks.iloc[i].title}')
        #     print(f'  - {talks.iloc[j].title}')
        #     D[i, j] = 0
    def __getitem__(self, ij):
        i, j = ij
        if i in self.missing or j in self.missing:
            return 0.0
        if i in self.talk_to_idx:
            i = self.talk_to_idx[i]
        else:
            if isinstance(i, Talk):
                t = self.talks.where(self.talks.email==i.email).last_valid_index()
                if t is None:
                    self.missing.add(i)
                    return 0.0
                self.talk_to_idx[i] = t
                i = t
        if j in self.talk_to_idx:
            j = self.talk_to_idx[j]
        else:
            if isinstance(j, Talk):
                t = self.talks.where(self.talks.email==j.email).last_valid_index()
                if t is None:
                    self.missing.add(j)
                    return 0.0
                self.talk_to_idx[j] = t
                j = t
        return self.D[i, j]


class JaccardDistance:
    def __init__(self, conf):
        self.J = defaultdict(float)
        # compute Jaccard similarity between talks
        participants_interested_in = defaultdict(set)
        for p in conf.participants:
            for t in p.preferences:
                participants_interested_in[t].add(p)
        for t1 in conf.talks:
            for t2 in conf.talks:
                I = len(participants_interested_in[t1] & participants_interested_in[t2])
                U = len(participants_interested_in[t1] | participants_interested_in[t2])
                if U:
                    self.J[t1, t2] = I/U
    def __getitem__(self, ij):
        return self.J[ij]


class SumDistance:
    def __init__(self, d1, d2):
        self.d1 = d1
        self.d2 = d2
    def __getitem__(self, item):
        return self.d1[item]+self.d2[item]


if __name__=='__main__':
    td = TopicDistance()
    print(td[1,3])