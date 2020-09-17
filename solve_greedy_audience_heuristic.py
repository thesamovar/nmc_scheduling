#%% Imports
import numpy as np
import pickle
import mip
from collections import defaultdict
import time
from generate_synthetic_data import generate_synthetic_data
from scipy import stats

#%% Load data
start_time = time.time()

talks_per_hour = 2

data = pickle.load(open('times_and_prefs_10k.pickle', 'rb'))

# data = generate_synthetic_data(
#     num_participants=1000, num_talks=100, hours_per_day=8, num_days=2,
#     timezones=[(1/3, 0, 4), (1/3, 0, 8), (1/3, 4, 8)],
#     available_per_day=lambda mu: stats.binom(3, 1.0),
#     clusters=1, prob_within_cluster=.25, prob_outside_cluster=1.0,
#     talk_interestingness_dist=stats.uniform(1.0, 0.0),
#     )

# data = generate_synthetic_data(
#         num_participants = 10_000,
#         num_talks = 1_000,
#         num_days = 5,
#         hours_per_day = 24,
#         timezones = [(0.25, 0, 10), (0.25, 4, 14), (0.25, 8, 18), (0.25, 14, 28)],
#         mean_available_per_day = stats.uniform(0, 10),
#         available_per_day = lambda mu: stats.binom(10., mu/10.),
#         clusters = 10,
#         prob_within_cluster = 0.3,
#         prob_outside_cluster = 0.03,
#         talk_interestingness_dist = stats.uniform(0, 1),
#         )

free_times = data['free_times']
prefs = data['prefs']
talk_clusters = data['talk_clusters']
talk_interestingness = data['talk_interestingness']
num_talks = data['num_talks']
num_times = data['num_times']*talks_per_hour
num_participants = len(prefs)
print(f'Finished loading data. {int(time.time()-start_time)}s')

#%% Generate matrices
I = defaultdict(int)
A = defaultdict(int)
F = defaultdict(int)
for p, f in enumerate(free_times):
    for s in f:
        for s in range(talks_per_hour*s, talks_per_hour*(s+1)):
            A[p, s] = 1
for t, f in enumerate(free_times[:num_talks]):
    for s in f:
        for s in range(talks_per_hour*s, talks_per_hour*(s+1)):
            F[t, s] = 1
for p, interest in enumerate(prefs):
    for t in interest:
        I[p, t] = 1
print(f'Finished setting up sparse matrices. {int(time.time()-start_time)}s')
print(f'  A has {len(A)} entries.')
print(f'  F has {len(F)} entries.')
print(f'  I has {len(I)} entries.')

#%% Greedy solution
# First, compute how popular each talk
popularity = defaultdict(int)
for (p, t) in I.keys():
    popularity[t] += 1
# Sort talks in order of popularity
popularity = list(popularity.items())
popularity.sort(key=lambda x: x[1], reverse=True)

# now go through talks and assign them to their best slot, then eliminate
# availability for participants for that slot
RA = A.copy()
hours = 0
talk_assignment = dict()
participant_schedule = defaultdict(list)
for t, _ in popularity:
    audience_total = defaultdict(int)
    audience = defaultdict(list)
    for p, s in RA.keys():
        if (p, t) in I and (t, s) in F:
            audience_total[s] += 1
            audience[s].append(p)
    if audience_total:
        s, _ = max(audience_total.items(), key=lambda x: x[1])
        talk_assignment[t] = s
        P = audience[s]
        hours += len(P)/talks_per_hour
        for p in P:
            participant_schedule[p].append((t, s))
            del RA[p, s]

#%% Some stats on the found solution
print(f"Found solution with {hours} total hours, {len(talk_assignment)} talks assigned of {num_talks}.")
tracks = defaultdict(int)
for (t, s) in talk_assignment.items():
    tracks[s] += 1
max_tracks = max(tracks.values())
print(f"Maximum number of tracks is {max_tracks}")

#%% verify found solution
for t, s in talk_assignment.items():
    if (t, s) not in F:
        raise ValueError((t, s))
for p, sched in participant_schedule.items():
    for (t, s) in sched:
        if (t, s) not in F or (p, t) not in I or (p, s) not in A:
            raise ValueError(p, t, s, sched)
