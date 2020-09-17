#%% Imports
import numpy as np
import pickle
import mip

#%% Load data
data = pickle.load(open('times_and_prefs_1k.pickle', 'rb'))
free_times = data['free_times']
prefs = data['prefs']
talk_clusters = data['talk_clusters']
talk_interestingness = data['talk_interestingness']
num_talks = data['num_talks']
num_times = data['num_times']
num_participants = len(prefs)

#%% Generate matrices
I = np.zeros((num_participants, num_talks), dtype=int)
A = np.zeros((num_participants, num_times), dtype=int)
F = np.zeros((num_talks, num_times), dtype=int)
for p, f in enumerate(free_times):
    A[p, f] = 1
for t, f in enumerate(free_times[:num_talks]):
    F[t, f] = 1
for p, interest in enumerate(prefs):
    I[p, interest] = 1

#%% Run analysis
model = mip.Model()

# Add decision variables
S = [[model.add_var(f'S({t},{s})', var_type=mip.BINARY) for s in range(num_times)] for t in range(num_talks)]
V = [[[model.add_var(f'V({p},{t},{s})', var_type=mip.BINARY) for s in range(num_times)] for t in range(num_talks)] for p in range(num_participants)]

# Add constraints
# only assign a talk to one slot
for t in range(num_talks):
    model += mip.xsum(S[t][s] for s in range(num_times))<=1#, f"talk_to_slot({t})"
# only assign talks to free slots
for s in range(num_times):
    for t in range(num_talks):
        model += S[t][s]<=F[t, s]
# viewers only watch talks in given slots if they're available and interested
for p in range(num_participants):
    for t in range(num_talks):
        for s in range(num_times):
            model += V[p][t][s] <= S[t][s]
            model += V[p][t][s] <= A[p, s]
            model += V[p][t][s] <= I[p, t]
# can only watch at most one talk per slot
for p in range(num_participants):
    for s in range(num_times):
        model += mip.xsum(V[p][t][s] for t in range(num_talks))<=1

# Add objective
model.objective = mip.maximize(mip.xsum(V[p][t][s] for p in range(num_participants) for t in range(num_talks) for s in range(num_times)))

#%% Solve it
model.optimize()

#%% Show the solution
print(f'Number of watch hours is {model.objective_value}')
for t in range(num_talks):
    for s in range(num_times):
        if S[t][s].x:
            break
    print(f'Assign talk {t} to slot {s}')
    can_watch = []
    for p in range(num_participants):
        if V[p][t][s].x:
            can_watch.append(p)
    print(f'   Participants that can watch: {can_watch}')