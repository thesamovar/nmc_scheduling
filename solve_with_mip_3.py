#%% Imports
import numpy as np
import pickle
import mip
from collections import defaultdict
import time
from generate_synthetic_data import generate_synthetic_data
from scipy import stats
from solve_greedy_audience_heuristic import greedy_solution

#%% Load data
start_time = time.time()

talks_per_hour = 1

data = pickle.load(open('times_and_prefs_10k.pickle', 'rb'))

# data = generate_synthetic_data(
#     num_participants=1000, num_talks=100, hours_per_day=8, num_days=2,
#     timezones=[(1/3, 0, 4), (1/3, 0, 8), (1/3, 4, 8)],
#     available_per_day=lambda mu: stats.binom(3, 1.0),
#     clusters=1, prob_within_cluster=.25, prob_outside_cluster=1.0,
#     talk_interestingness_dist=stats.uniform(1.0, 0.0),
#     )

free_times = data['free_times']
prefs = data['prefs']
talk_clusters = data['talk_clusters']
talk_interestingness = data['talk_interestingness']
num_talks = data['num_talks']
num_times = data['num_times']*talks_per_hour
num_participants = len(prefs)
print(f'Finished loading data. {int(time.time()-start_time)}s')

mean_hours_free = np.mean([len(f) for f in free_times])
mean_num_interested = np.mean([len(pref) for pref in prefs])

ideal_hours = 0
for p in range(num_participants):
    ideal_hours += min(len(free_times[p]), len(prefs[p])/talks_per_hour)

print("Mean hours free", mean_hours_free)
print("Max hours per participant", ideal_hours/num_participants)
print("Max total hours", ideal_hours)
print("Mean num interested", mean_num_interested)

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
model.objective = mip.maximize(mip.xsum(V.values())/talks_per_hour)

#%% Generate greedy solution to initialise
print("Start finding greedy solution.")
greedy_talk_assignment, greedy_participant_schedule = greedy_solution(data, talks_per_hour=talks_per_hour, verbose=True)
init_solution = []
for t, s in greedy_talk_assignment.items():
    init_solution.append((S[t, s], 1))
for p, sched in greedy_participant_schedule.items():
    for (t, s) in sched:
        init_solution.append((V[p, t, s], 1))
model.start = init_solution
print(f"Finished finding greedy solution. {int(time.time()-start_time)}s")

#%% Solve it
model.optimize(max_seconds=60*5)

#%% Show the solution
talk_assignment = dict()
participant_schedule = defaultdict(list)
for t in range(num_talks):
    for s in range(num_times):
        if (t, s) in S and S[t, s].x:
            talk_assignment[t] = s
            break
    print(f'Assign talk {t} to slot {s}')
    can_watch = []
    for p in range(num_participants):
        if (p, t, s) in V and V[p, t, s].x:
            can_watch.append(p)
            participant_schedule[p].append((t, s))
    print(f'   Participants that can watch: {can_watch}')

print(f'Number of watch hours is {int(model.objective_value)} of max possible {ideal_hours}, or {100.0*model.objective_value/ideal_hours:.1f}%')
print(f"Talks assigned: {len(talk_assignment)} of {num_talks}.")
tracks = defaultdict(int)
for (t, s) in talk_assignment.items():
    tracks[s] += 1
max_tracks = max(tracks.values())
print(f"Maximum number of tracks is {max_tracks}")

# verify found solution
for t, s in talk_assignment.items():
    if (t, s) not in F:
        raise ValueError((t, s))
for p, sched in participant_schedule.items():
    for (t, s) in sched:
        if (t, s) not in F or (p, t) not in I or (p, s) not in A:
            raise ValueError(p, t, s, sched)

print(f"Finished, elapsed time is {int(time.time()-start_time)}s")