#%% Imports
import numpy as np
from scipy import stats
from random import shuffle
import matplotlib.pyplot as plt
import pickle
import os

#%% Parameters
# Participants
num_participants = 10_000

# Presenters, consider the first participants to be presenters
num_talks = 1_000

# Time slots and zones
num_days = 5
num_times = num_days*24
# (p, start_time, end_time)
timezones = [(0.25, 0, 10), (0.25, 4, 14), (0.25, 8, 18), (0.25, 14, 28)]
mean_available_per_day = stats.uniform(0, 10)
available_per_day = lambda mu: stats.binom(10., mu/10.)

# how participants interests cluster, they choose talks within their cluster with a
# fixed probability, and similarly outside cluster (different probability).
clusters = 10
prob_within_cluster = 0.3
prob_outside_cluster = 0.03
talk_interestingness_dist = stats.uniform(0, 1)

#%% Generate free times for all participants
timezone_probs, _, _ = zip(*timezones)
participant_timezones = stats.rv_discrete(values=(np.arange(len(timezones)), timezone_probs)).rvs(size=num_participants)
free_times = []
for i in range(num_participants):
    _, start, end = timezones[participant_timezones[i]]
    mu = mean_available_per_day.rvs(size=1)
    
    # Disallow cases where talks have zero availability
    n_avail = 0
    while np.sum(n_avail) == 0:
        n_avail = available_per_day(mu).rvs(size=num_days)

    I = np.arange(start, end) % 24
    free = []
    for day in range(num_days):
        shuffle(I)
        free.extend(day*24+I[:n_avail[day]])
    free_times.append(np.array(free, dtype=int))


#%% Generate talks and preferences
participant_clusters = stats.randint(0, clusters).rvs(size=num_participants)
talk_clusters = participant_clusters[:num_talks] # assume speakers prefer their own topic
talk_interestingness = talk_interestingness_dist.rvs(size=num_talks)
prefs = []
for i in range(num_participants):
    p = np.full(num_talks, prob_outside_cluster)
    p[participant_clusters[i]==talk_clusters] = prob_within_cluster
    p = p*talk_interestingness
    preferred = (stats.uniform.rvs(size=num_talks)<p).nonzero()
    prefs.append(preferred)

#%% Quickly generate some plots/stats
interested_count = np.zeros(num_talks, dtype=int)
for p in prefs:
    interested_count[p] += 1
plt.figure()
plt.hist(interested_count, bins=20)

available_count = np.zeros(num_times, dtype=int)
for free in free_times:
    available_count[free] += 1
plt.figure()
plt.bar(np.arange(num_times), available_count, width=1.0)
plt.xlim(0, 24)
#plt.savefig('free_times.png')

#%% Dump data to pickle file
data = {'free_times': free_times, 
        'prefs': prefs, 
        'talk_clusters': talk_clusters, 
        'talk_interestingness': talk_interestingness}
with open('times_and_prefs.pickle', 'wb') as f:
    pickle.dump(data, f)