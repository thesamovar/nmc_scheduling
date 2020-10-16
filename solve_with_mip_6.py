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

talks_per_hour = 3

data = pickle.load(open('times_and_prefs_2k_500.pickle', 'rb'))

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


#%% Generate greedy solution to initialise
print("Start finding greedy solution.")
greedy_talk_assignment, greedy_participant_schedule = greedy_solution(data, talks_per_hour=talks_per_hour, verbose=True)
print(f"Finished finding greedy solution. {int(time.time()-start_time)}s")

# use the greedy solution to determine the number of tracks
tracks = defaultdict(int)
for (t, s) in greedy_talk_assignment.items():
    tracks[s] += 1
num_tracks = max(tracks.values())
print(f"Maximum number of tracks is {num_tracks}")

#S_gre = defaultdict(int)
#V_gre = defaultdict(int)
S_gre = {}
V_gre = {}
for s in range(num_times):
    k = 0
    for t in range(num_talks):
        if (t,s) in greedy_talk_assignment.items():
            S_gre[t,k,s] = 1
            k += 1
            if k > num_tracks:
                raise ValueError("k > num_tracks")            
for p, sched in greedy_participant_schedule.items():
    for (t, s) in sched:
        for k in range(num_tracks):
            if (t,k,s) in S_gre:
                V_gre[p,k,s] = 1

#%% Generate matrices
#I = defaultdict(int)
#A = defaultdict(int)
#F = defaultdict(int)
I = {}
A = {}
F = {}
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


# Set the optimization iteration
opt_iter = 3;
for opt_count in range(opt_iter):
    #%% Lower-level optimization of V_low[p,k,s], where k is the track index
    model_low = mip.Model()
    # Add constant matrices for lower-level optimization
    #S_low = defaultdict(int)
    S_low = {}
    # for the first iteration, use the greedy solution S_gre[t,k,s] for an estimate of S_up[t,k,s]
    if opt_count == 0:
        for (t,s) in F:
            for k in range(num_tracks):
                if (t,k,s) in S_gre:
                    S_low[t,k,s] = 1
    else:
        for (t,k,s) in S_up:
            if S_up[t,k,s].x:
                S_low[t,k,s] = 1

    # Set pay-off matrix
    #P_low = defaultdict(int)
    P_low = {}
    for (p,t) in I:
        for k in range(num_tracks):
            for s in range(num_times):
                if (t,k,s) in S_low:
                    P_low[p,k,s] = 1

    print(f'Finished setting up constant matrices for lower-level. {int(time.time()-start_time)}s')
    print(f'  S_low has {len(S_low)} entries.')
    print(f'  P_low has {len(P_low)} entries.')  
       
    # Add decision variables
    V_low = {}
    for (p,t) in I:
        for s in range(num_times):
            if (p,s) in A and (p,t) in I and (t,s) in F:
                for k in range(num_tracks):
                    V_low[p,k,s] = model_low.add_var(f'V_low({p},{k},{s})', var_type=mip.BINARY)

    print(f"Finished setting up decision variables for lower-level. {int(time.time()-start_time)}s")
    print(f"  V_low: {len(V_low)} decision variables")


    # Add constraints
    nconstraints_low = 0

    # a viewer can only go to one (track,slot)
    for (p,s) in A:
        model_low += mip.xsum(V_low[p,k,s] for k in range(num_tracks) if (p,k,s) in V_low)<=1
        nconstraints_low += 1

    print(f"Finished setting up constraints for lower-level. {int(time.time()-start_time)}s")
    print(f"  Number of constraints for lower-level is {nconstraints_low}")

    # Add objective
    model_low.objective = mip.maximize(mip.xsum(P_low[p,k,s]*V_low[p,k,s] for (p,k,s) in V_low if (p,k,s) in P_low)/talks_per_hour)

    #%% Now solve it with ILP to get best integer solution
    opt_status_low = model_low.optimize(relax=False)
    print(f"Lower-level optimization: {opt_status_low}")
    print(f"objective = {model_low.objective_value}\n")

    #%% Upper-level optimization of S_up[t,k,s], where k is the track index
    model_up = mip.Model()
    # Add constant matrices for upper-level optimization
    #V_up = defaultdict(int)
    V_up = {}
    for (p,k,s) in V_low:
        if V_low[p,k,s].x:
            V_up[p,k,s] = 1     

    # Set pay-off matrix
    #P_up = defaultdict(int)
    P_up = {}
    for k in range(num_tracks):
        for s in range(num_times):
            for (p,t) in I:
                if (p,k,s) in V_up:
                    if (t,k,s) in P_up:
                        P_up[t,k,s] += 1
                    else:
                        P_up[t,k,s] = 1

    print(f'Finished setting up constant matrices for upper-level. {int(time.time()-start_time)}s')
    print(f'  V_up has {len(V_up)} entries.')
    print(f'  P_up has {len(P_up)} entries.')  
          
    # Add decision variables
    S_up = {}
    for (t,s) in F:
        for k in range(num_tracks):
            S_up[t,k,s] = model_up.add_var(f'S_up({t},{k},{s})', var_type=mip.BINARY)

    print(f"Finished setting up decision variables for upper-level. {int(time.time()-start_time)}s")
    print(f"  S_up: {len(S_up)} decision variables")

    # Add constraints
    nconstraints_up = 0
    # only assign a talk to one slot on one track
    for t in range(num_talks):
        model_up += mip.xsum( mip.xsum(S_up[t,k,s] for k in range(num_tracks) if (t,k,s) in S_up) for s in range(num_times) ) <= 1
        nconstraints_up += 1
    # each (track,slot) can only have at most one talk assigned
    for k in range(num_tracks):
        for s in range(num_times):
            model_up += mip.xsum(S_up[t,k,s] for t in range(num_talks) if (t,k,s) in S_up) <= 1
            nconstraints_up += 1

    print(f"Finished setting up constraints for upper-level. {int(time.time()-start_time)}s")
    print(f"  Number of constraints for upper-level is {nconstraints_up}")

    # Add objective
    model_up.objective = mip.maximize(mip.xsum(P_up[t,k,s]*S_up[t,k,s] for (t,k,s) in S_up if (t,k,s) in P_up)/talks_per_hour)

    #%% Now solve it with ILP to get best integer solution
    opt_status_up = model_up.optimize(relax=False)
    print(f"Upper-level optimization: {opt_status_up}")
    print(f"objective = {model_up.objective_value}")

#%% Show the solution
# Restore S_up[t,k,s] and V_low[p,k,s] back to S[t,s] and V[p,t,s]
#S = defaultdict(int)
S = {}
#V = defaultdict(int)
V = {}
for (p,k,s) in V_low:
    if V_low[p,k,s].x:
        for t in range(num_talks):
            if (t,k,s) in S_up:
                if S_up[t,k,s].x:
                    S[t,s] = 1
                    V[p,t,s] = 1
                    
# The rest are the same as the exact method
talk_assignment = dict()
participant_schedule = defaultdict(list)
watch_hours_on_schedule = 0
for t in range(num_talks):
    for s in range(num_times):
        if (t, s) in S and S[t, s]:
            talk_assignment[t] = s
            break
    #print(f'Assign talk {t} to slot {s}')
    can_watch = []
    for p in range(num_participants):
        if (p, t, s) in V and V[p, t, s]:
            can_watch.append(p)
            participant_schedule[p].append((t, s))
            watch_hours_on_schedule += 1/talks_per_hour
    #print(f'   Participants that can watch: {can_watch}')

print(f'Number of watch hours on schedule is {watch_hours_on_schedule:1f}')
print(f'Number of watch hours is {model_up.objective_value:.1f} of max possible {ideal_hours:1f}, or {100.0*model_up.objective_value/ideal_hours:.1f}%')
#print(f'   - watched hours is {100*model.objective_value/relaxed_opt:.1f}% of relaxed relaxed optimum {relaxed_opt}')
print(f"Talks assigned: {len(talk_assignment)} of {num_talks}.")
tracks = defaultdict(int)
for (t, s) in talk_assignment.items():
    tracks[s] += 1
max_tracks = max(tracks.values())
print(f"Maximum number of tracks is {max_tracks}")

#%% verify found solution
violations = []
overcounting_hours = 0
for t, s in talk_assignment.items():
    if (t, s) not in F:
        violations.append(f"Violation, talk assigned to unavailable slot {t}, {s}")
for p, sched in participant_schedule.items():
    _, curtimes = zip(*sched)
    if len(np.unique(curtimes))!=len(curtimes):
        overcounting_hours += (len(curtimes)-len(np.unique(curtimes)))/talks_per_hour
        violations.append(f"Violation, participant assigned to multiple talks at same slot: {p}, {curtimes}")
    for (t, s) in sched:
        if (t, s) not in F or (p, t) not in I or (p, s) not in A:
            violations.append(f"Violation, participant assigned incorrectly {p}, {t}, {s}")
if violations:
    print(f"Overcounted hours: {overcounting_hours}")
print(f"Finished, elapsed time is {int(time.time()-start_time)}s")
# %%
