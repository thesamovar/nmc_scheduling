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

#data = pickle.load(open('times_and_prefs_10k.pickle', 'rb'))

data = generate_synthetic_data(
    num_participants=1000, num_talks=100, hours_per_day=8, num_days=2,
    timezones=[(1/3, 0, 4), (1/3, 0, 8), (1/3, 4, 8)],
    available_per_day=lambda mu: stats.binom(3, 1.0),
    clusters=1, prob_within_cluster=.25, prob_outside_cluster=1.0,
    talk_interestingness_dist=stats.uniform(1.0, 0.0),
    )

free_times = data['free_times']
prefs = data['prefs']
talk_clusters = data['talk_clusters']
talk_interestingness = data['talk_interestingness']
num_talks = data['num_talks']
num_times = data['num_times']
num_participants = len(prefs)
print(f'Finished loading data. {int(time.time()-start_time)}s')

#%% Generate matrices
I = defaultdict(int)
A = defaultdict(int)
F = defaultdict(int)
for p, f in enumerate(free_times):
    for s in f:
        A[p, s] = 1
for t, f in enumerate(free_times[:num_talks]):
    for s in f:
        F[t, s] = 1
for p, interest in enumerate(prefs):
    for t in interest:
        I[p, t] = 1
print(f'Finished setting up sparse matrices. {int(time.time()-start_time)}s')
print(f'  A has {len(A)} entries.')
print(f'  F has {len(F)} entries.')
print(f'  I has {len(I)} entries.')

#%% Setup model
model = mip.Model()

# Add decision variables
S = {}
for (t, s) in F.keys():
    S[t, s] = model.add_var(f'S({t},{s})', var_type=mip.BINARY)
V = {}
for (p, t) in I.keys():
    for s in range(num_times):
        # ensure V_pts<=F_ts, A_ps, I_pt
        if A[p, s] and I[p, t] and F[t, s]:
            V[p, t, s] = model.add_var(f'V({p},{t},{s})', var_type=mip.BINARY)

print(f"Finished setting up decision variables. {int(time.time()-start_time)}s")
print(f"  S: {len(S)}, V: {len(V)}, total {len(S)+len(V)}")

# Add constraints
# only assign a viewer to a talk in a given slot if it has been assigned there
nconstraints = 0
for (p, t, s) in V.keys():
    model += V[p, t, s]<=S[t, s]
    #model.add_lazy_constr(V[p, t, s]<=S[t, s])
    nconstraints += 1
# only assign a talk to one slot
for t in range(num_talks):
    model += mip.xsum(S[t, s] for s in range(num_times) if (t, s) in S)<=1
    #model.add_lazy_constr(mip.xsum(S[t, s] for s in range(num_times) if (t, s) in S)<=1)
    nconstraints += 1
# can only watch at most one talk per slot
for (p, s) in A.keys():
    model += mip.xsum(V[p, t, s] for t in range(num_talks) if (p, t, s) in V)<=1
    #model.add_lazy_constr(mip.xsum(V[p, t, s] for t in range(num_talks) if (p, t, s) in V)<=1)
    nconstraints += 1

print(f"Finished setting up constraints. {int(time.time()-start_time)}s")
print(f"  Number of constraints is {nconstraints}")

# Add objective
model.objective = mip.maximize(mip.xsum(V.values()))

#%% Solve it
model.optimize()

#%% Show the solution
for t in range(num_talks):
    for s in range(num_times):
        if (t, s) in S and S[t, s].x:
            break
    print(f'Assign talk {t} to slot {s}')
    can_watch = []
    for p in range(num_participants):
        if (p, t, s) in V and V[p, t, s].x:
            can_watch.append(p)
    print(f'   Participants that can watch: {can_watch}')

max_obj = len(A)
print(f'Number of watch hours is {int(model.objective_value)} of max possible {max_obj}, or {100.0*model.objective_value/max_obj:.1f}%')
print(f"Finished, elapsed time is {int(time.time()-start_time)}s")