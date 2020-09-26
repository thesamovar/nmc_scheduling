'''
Automatic scheduler for neuromatch conference
'''

import numpy as np
import mip
from collections import defaultdict
import time
from conference import Conference, Talk, Participant, load_nmc3, load_synthetic

def generate_schedule(submissions, participants):
    pass

if __name__=='__main__':
    import pickle
    start_time = time.time()
    # load and convert synthetic data
    data = pickle.load(open('times_and_prefs_2k_500.pickle', 'rb'))
    free_times = data['free_times']
    prefs = data['prefs']
    talk_clusters = data['talk_clusters']
    talk_interestingness = data['talk_interestingness']
    num_talks = data['num_talks']
    num_times = data['num_times']*talks_per_hour
    num_participants = len(prefs)
    submissions = 
    print(f'Finished loading data. {int(time.time()-start_time)}s')
