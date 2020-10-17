#%%
import numpy as np
from collections import defaultdict
import time
from conference import Conference, Talk, Participant, load_nmc3, load_synthetic
import datetime
from schedule_writer import html_schedule_dump, html_participant_dump
import dateutil.parser
from topic_distance import TopicDistance, JaccardDistance, SumDistance
import itertools
import numba
import mip
import pickle
import os

def compute_participant_availability_distributions(conf):
    traditional=defaultdict(int)
    interactive_strict=defaultdict(int)
    interactive_loose=defaultdict(int)
    for participant in conf.participants:
        for avail in participant.available:
            traditional[avail] += 1
            interactive_loose[avail] += 1
            interactive_loose[avail-1] += 1
            if avail+1 in participant.available.available:
                interactive_strict[avail] += 1
    return dict(
        traditional=traditional,
        interactive_strict=interactive_strict,
        interactive_loose=interactive_loose,
        )

def estimate_audience_for_sessions(conf, session_assignment, potential_audience, I):
    sessions_at_time = defaultdict(list)
    for talks, h in session_assignment.items():
        sessions_at_time[h].append(talks)
    audience = defaultdict(float)
    for h, sessions in sessions_at_time.items():
        for p in conf.participants:
            fraction = defaultdict(float)
            for session in sessions:
                for talk in talks:
                    if I[p, talk]:
                        fraction[session] += 1
            if fraction:
                fsum = sum(fraction.values())
                for session, f in fraction.items():
                    audience[session] += f/fsum
    return audience, sum(audience.values())

# Greedy scheduler
def generate_greedy_schedule(conf, max_tracks=7, estimated_audience=10_000,
                             talks_per_hour=3,
                             blocked_times=None,
                             min_shared_times=2,
                             ):
    start_time = time.time()
    free_time_distributions = compute_participant_availability_distributions(conf)
    freedist = free_time_distributions['traditional']
    scaling = estimated_audience/conf.num_participants
    interactive_talks = [i for i, talk in enumerate(conf.talks) if talk.talk_format=="Interactive talk"]
    traditional_talks = [i for i, talk in enumerate(conf.talks) if talk.talk_format=="Traditional talk"]
    num_hours = conf.num_hours

    # handle blocked times
    if blocked_times is None:
        blocked_times = set([])
    new_blocked_times = set([])
    for t in blocked_times:
        if isinstance(t, str): # convert from time format to datetime
            t = dateutil.parser.parse(t)
        if not isinstance(t, int):
            t = int((t-conf.start_time).total_seconds())//(60*60)
        new_blocked_times.add(t)
    blocked_times = new_blocked_times

    # Generate input matrices
    talk_available_times = defaultdict(set)
    potential_audience = defaultdict(set)
    for p in conf.participants:
        for t in p.preferences:
            if not isinstance(t, Talk):
                t = conf.talks[t]
            potential_audience[t].add(p)
    for t, talk in enumerate(conf.talks):
        talk._idx = t
        for s in talk.available:
            if s in blocked_times:
                continue
            if s<0 or s>=conf.num_hours:
                continue
            talk_available_times[talk].add(s)
    audience_overlap = defaultdict(float)
    schedule_overlap = defaultdict(float)
    with_shared_schedule = defaultdict(set)
    with_shared_audience = defaultdict(set)
    with_shared_schedule_and_audience = defaultdict(set)
    for talk1 in conf.talks:
        for talk2 in conf.talks:
            if talk1.talk_format!=talk2.talk_format:
                continue
            audience_overlap[talk1, talk2] = potential_audience[talk1].intersection(potential_audience[talk2])
            schedule_overlap[talk1, talk2] = talk_available_times[talk1].intersection(talk_available_times[talk2])
            if talk1 is not talk2:
                if audience_overlap[talk1, talk2]:
                    with_shared_audience[talk1].add(talk2)
                if len(schedule_overlap[talk1, talk2])>=min_shared_times:
                    with_shared_schedule[talk1].add(talk2)
                if audience_overlap[talk1, talk2] and len(schedule_overlap[talk1, talk2])>=min_shared_times:
                    with_shared_schedule_and_audience[talk1].add(talk2)
    print(f'Finished precomputations. {int(time.time()-start_time)}s')
    print(f'Mean number of overlapping talks and audience: {np.mean([len(cotalks) for talk, cotalks in with_shared_schedule_and_audience.items()]):.1f}')

    # find best session
    conf.talk_assignment = defaultdict(int)
    conf.participant_schedule = {}
    conf.track_assignment = defaultdict(int)
    next_track = defaultdict(int)
    alternate_times = {}
    session_assignment = {}
    sessions_file = open('sessions_generated.txt', 'wb')
    possible_sessions = []
    #pairwise = TopicDistance()
    #pairwise = JaccardDistance(conf)
    pairwise = SumDistance(TopicDistance(), JaccardDistance(conf))
    def session_metric(talk1, talk2, talk3):
        return pairwise[talk1, talk2]+pairwise[talk2, talk3]+pairwise[talk1, talk3]
        #return len(audience_overlap[talk1, talk2].intersection(potential_audience[talk3]))
    for talk1 in conf.talks:
        for talk2 in with_shared_schedule[talk1]:
            if talk2._idx<=talk1._idx:
                continue
            if len(with_shared_schedule[talk1])<len(with_shared_schedule[talk2]):
                next_talks = with_shared_schedule[talk1]
            else:
                next_talks = with_shared_schedule[talk2]
            for talk3 in next_talks:
                if talk3._idx<=talk2._idx:
                    continue
                if talk3 is talk1 or talk3 is talk2:
                    continue
                times = schedule_overlap[talk1, talk2].intersection(talk_available_times[talk3])
                if len(times)<min_shared_times:
                    continue
                metric = session_metric(talk1, talk2, talk3)
                possible_sessions.append((talk1, talk2, talk3, metric))
    possible_sessions.sort(key=lambda x: x[3], reverse=True)
    already_assigned = set()
    all_metrics = []
    for talk1, talk2, talk3, best_metric in possible_sessions:
        if talk1 in already_assigned or talk2 in already_assigned or talk3 in already_assigned:
            continue
        all_metrics.append(best_metric)
        best_trio = (talk1, talk2, talk3)
        best_times = schedule_overlap[talk1, talk2].intersection(talk_available_times[talk3])
        msg = f'Metric={best_metric}, number of times={len(best_times)}\n'
        for t in best_trio:
            msg += ' - '+t.title[:150]+'\n'
        sessions_file.write(msg.encode('UTF-8'))
        alternate_times[best_trio] = best_times
        best_trio[0].scheduling_message = f"Session metric: {best_metric}"
        # remove those talks
        for talk in best_trio:
            already_assigned.add(talk)
    print(f'Mean metric: {np.mean(all_metrics):.3f}')
    # print('Max num tracks:', max(next_track.values()))
    print('Unscheduled talks:', len(conf.talks)-len(already_assigned))
    print(f'Finished. {int(time.time()-start_time)}s')
    conf.alternate_times = alternate_times
    conf.potential_audience = potential_audience
    return conf

#%%

def assign_sessions(conf, max_track_vector, estimated_audience=10_000, relax=True, model_kwds=None):
#%%
    start_time = time.time()
    free_time_distributions = compute_participant_availability_distributions(conf)
    freedist = free_time_distributions['traditional']
    sessions = conf.alternate_times
    # set up input matrices
    W = defaultdict(float)
    A = defaultdict(int)
    all_times = set()
    for p in conf.participants:
        for s in sessions.keys():
            for talk in s:
                if talk in p.preferences:
                    W[p, s] += 1/3
    for s, times in sessions.items():
        for t in times:
            if t<0 or t>=conf.num_hours:
                continue
            all_times.add(t)
            A[s, t] = 1
    print(f'Finished setting up input matrices. {int(time.time()-start_time)}s')
    print(f'  A has {len(A)} entries.')
    print(f'  W has {len(W)} entries.')
    # create the model
    model = mip.Model()
    # Add decision variables
    S = {}
    V = {}
    for (s, t) in A.keys():
        S[s, t] = model.add_var(var_type=mip.BINARY)
        for p in conf.participants:
            if (p, s) in W:
                V[p, s, t] = model.add_var(var_type=mip.BINARY)
    print(f"Finished setting up decision variables. {int(time.time()-start_time)}s")
    print(f"  S: {len(S)}, V: {len(V)}, total {len(S)+len(V)}")
    # Add constraints
    for s in sessions.keys():
        model += mip.xsum(S[s, t] for t in all_times if (s, t) in S)==1 # only assign talk to 1 time
    for t in all_times:
        model += mip.xsum(S[s, t] for s in sessions.keys() if (s, t) in S)<=max_track_vector[t] # only use at most max tracks
    for p in conf.participants:
        for t in all_times:
            model += mip.xsum(V[p, s, t] for s in sessions.keys() if (p, s, t) in V)<=1 # choose at most one slot to view at each time for each participant
    for p, s, t in V.keys():
        model += V[p, s, t]<=S[s, t] # can only watch session t at time t if session s is assigned to that time
    print(f"Finished setting up constraints. {int(time.time()-start_time)}s")
    print(f"  Number of constraints is {len(model.constrs)}")        
    # Add objective
    watch_hours = mip.xsum(V[p, s, t]*W[p, s]*freedist[t] for (p, s, t) in V.keys())
    #L1 = mip.xsum(V.values())+mip.xsum(S.values())
    objfun = watch_hours#-L1
    model.objective = mip.maximize(objfun)
    # Solve it with relaxed model first to get upper bound
    # model.emphasis = 1 # feasibility first
    # model.max_mip_gap = 0.1 # allow a solution within 10% of optimal
    # model.max_seconds = 60*60
    # model.max_mip_gap = 0.2
    # model.max_seconds = 10*60
    # model.cutoff = -2e6 # not sure if this is a good idea or not...
    if model_kwds is not None:
        for k, v in model_kwds.items():
            setattr(model, k, v)
    opt_status = model.optimize(relax=relax)
    print(opt_status)
    print(f'Expected watch hours assuming audience of {estimated_audience} is {model.objective_value*estimated_audience/conf.num_participants**2}')
    # plt.hist([v.x for v in V.values()], alpha=0.7)
    # plt.hist([v.x for v in S.values()], alpha=0.7)
    # plt.show()
#%%
    # Convert it into a schedule (allow for possibility it's not binary)
    next_track = defaultdict(int)
    ordered_assignments = sorted(S.keys(), key=lambda x: S[x].x, reverse=True)
    session_assignment = {}
    for (s, t) in ordered_assignments:
        if S[s, t].x==0:
            break
        if s in session_assignment:
            continue
        if next_track[t]>=max_track_vector[t]:
            continue
        session_assignment[s] = t
        next_track[t] += 1
    next_track = defaultdict(int)
    for s, t in session_assignment.items():
        for offset, talk in enumerate(s):
            conf.talk_assignment[talk] = 3*t+offset
            conf.track_assignment[talk] = next_track[t]
        next_track[t] += 1
    unassigned = set(sessions.keys()).difference(set(session_assignment.keys()))
    print(f'{len(unassigned)} unassigned sessions')
    # import importlib
    # import schedule_writer
    # importlib.reload(schedule_writer)
    # from schedule_writer import html_schedule_dump
    # html_schedule_dump(conf)
#%%


def greedy_assign_sessions(conf, max_tracks=7, estimated_audience=10_000):
    start_time = time.time()
    free_time_distributions = compute_participant_availability_distributions(conf)
    freedist = free_time_distributions['traditional']
    sessions = conf.alternate_times
    # set up input matrices
    W = defaultdict(float)
    A = defaultdict(int)
    all_times = set()
    potential_session_audience = defaultdict(set)
    for p in conf.participants:
        for s in sessions.keys():
            for talk in s:
                if talk in p.preferences:
                    W[p, s] += 1/3
                    potential_session_audience[s].add(p)
    for s, times in sessions.items():
        for t in times:
            if t<0 or t>conf.num_hours:
                continue
            all_times.add(t)
            A[s, t] = 1
    print(f'Finished setting up input matrices. {int(time.time()-start_time)}s')
    print(f'  A has {len(A)} entries.')
    print(f'  W has {len(W)} entries.')
    remaining_sessions = set(sessions.keys())
    available_audience = defaultdict(set)
    for p in conf.participants:
        for h in p.available.available:
            if h<0 or h>=conf.num_hours:
                continue
            available_audience[h].add(p)
    num_tracks = defaultdict(int)
    remaining_times = set(range(conf.num_hours))
    session_assignment = {}
    while remaining_sessions:
        try:
            session, h = max(((session, h) for session in remaining_sessions if len(sessions[session].intersection(remaining_times)) for h in sessions[session].intersection(remaining_times)), key=lambda x: len(available_audience[x[1]].intersection(potential_session_audience[x[0]])))
        except ValueError: # no schedulable slots left
            break
        session_assignment[session] = h
        remaining_sessions.remove(session)
        num_tracks[h] += 1
        if num_tracks[h]>=max_tracks:
            remaining_times.remove(h)
    next_track = defaultdict(int)
    for s, t in session_assignment.items():
        for offset, talk in enumerate(s):
            conf.talk_assignment[talk] = 3*t+offset
            conf.track_assignment[talk] = next_track[t]
        next_track[t] += 1
    unassigned = set(sessions.keys()).difference(set(session_assignment.keys()))
    print(f'{len(unassigned)} unassigned sessions')
    return conf


def slot_unscheduled(conf, blocked_times, max_track_vector):
    # There are now a number of unscheduled talks. Reschedule these talks
    # wherever they will fit.
    new_blocked_times = set([])
    for t in blocked_times:
        if isinstance(t, str): # convert from time format to datetime
            t = dateutil.parser.parse(t)
        if not isinstance(t, int):
            t = int((t-conf.start_time).total_seconds())//(60*60)
        new_blocked_times.add(t)

    avail_sessions = max_track_vector.copy()

    # Compute the number of sessions which can still be allotted.
    for t, track in conf.track_assignment.items():
        the_time = conf.talk_assignment[t] // 3
        avail_sessions[the_time] = min(avail_sessions[the_time], max_tracks - track - 1)
    
    for x in new_blocked_times:
        if x >= 0 and x < conf.num_hours:
            avail_sessions[x] = 0

    # Generate input matrices
    S = []
    unscheduled_talks = []
    for talk in conf.talks:
        if talk not in conf.talk_assignment:
            unscheduled_talks.append(talk)
            A = np.zeros(conf.num_hours)
            for s in talk.available:
                if s<0 or s>=conf.num_hours:
                    continue
                if avail_sessions[s] == 0:
                    continue 
                A[s] = 1
            S.append(A)

    S = np.array(S).copy()
    avail_sessions = avail_sessions.copy()
    triplets = []

    # Start with triplets, end with doublets
    rescheduled = 0
    for ns in [3, 2]:
        the_order = S.sum(1).argsort()
        for n in the_order:
            # Check if there is any way to make a triplet
            schedulable = S.sum(0)
            good_slots = (schedulable >= ns) & ([S[n, :].squeeze() > 0]) & (avail_sessions > 0)
            if good_slots.any():
                slot = np.where(good_slots[0])[0][0]
                targets = np.where(S[:, slot] > 0)[0]
                members = []
                for target in targets:
                    members.append((S[target, :].sum(), target))
                
                # Pick the ns members with the least availability and put them 
                # together.
                triplet = [x[1] for x in sorted(members)[:ns]]
                for tri in triplet:
                    S[tri, :] = 0
                    rescheduled += 1

                avail_sessions[slot] -= 1
                triplets.append((slot, max_tracks - avail_sessions[slot] - 1, triplet))
    
    for slot, track, triplet in triplets:
        for i, t in enumerate(triplet):
            talk = unscheduled_talks[t]
            conf.track_assignment[talk] = track
            conf.talk_assignment[talk] = slot * 3 + i

    print(f"=== Succesfully rescheduled {rescheduled}/{S.shape[0]} talks ===")
    print("The following talks are unscheduled")
    S = []
    for talk in conf.talks:
        if talk not in conf.talk_assignment:
            A = np.zeros(conf.num_hours)
            for s in talk.available:
                if s<0 or s>=conf.num_hours:
                    continue
                if avail_sessions[s] == 0:
                    continue 
                A[s] = 1
            S.append(A)
            print(talk.title)
            print(f'available {A.sum()} times')


def reserved_times_to_max_tracks(conf, reserved_times, max_tracks):
    max_track_vector = max_tracks * np.ones(conf.num_hours, dtype=np.int)
    for t in reserved_times:
        if isinstance(t, str): # convert from time format to datetime
            t = dateutil.parser.parse(t)
        if not isinstance(t, int):
            t = int((t-conf.start_time).total_seconds())//(60*60)
        max_track_vector[t] -= 1
    return max_track_vector

def dump_to_json(conf):
    schedule = collections.defaultdict(lambda: {})
    for t, assignment in conf.talk_assignment.items():
        track = conf.track_assignment[t]
        the_slot = conf.start_time + datetime.timedelta(hours=int(assignment) // 3) + datetime.timedelta(minutes=15 * int(assignment % 3))
        d1 = the_slot.strftime("%Y-%m-%dT%H:%M:%SZ")
        print(d1)
        the_dict = t.__dict__
        if 'available' in the_dict:
            del the_dict['available']
        if 'Unnamed: 0' in the_dict:
            del the_dict['Unnamed: 0']
        schedule[d1][f'track {track + 1}'] = the_dict
        
    with open('schedule.json', 'w') as f:
        json.dump(schedule, f)

if __name__=='__main__':
    import os, pickle
    import matplotlib.pyplot as plt
    # Generate the thematically coherent sessions that are scheduled below
    blocked_times = [
        # keynote times (8am UTC not used)
        '2020-10-26 00:00 UTC', '2020-10-26 07:00 UTC', '2020-10-26 14:00 UTC', '2020-10-26 15:00 UTC', '2020-10-26 19:00 UTC', '2020-10-26 20:00 UTC', '2020-10-26 23:00 UTC',
        '2020-10-27 00:00 UTC', '2020-10-27 07:00 UTC', '2020-10-27 14:00 UTC', '2020-10-27 15:00 UTC', '2020-10-27 19:00 UTC', '2020-10-27 20:00 UTC', '2020-10-27 23:00 UTC',
        '2020-10-28 00:00 UTC', '2020-10-28 07:00 UTC', '2020-10-28 14:00 UTC', '2020-10-28 15:00 UTC', '2020-10-28 19:00 UTC', '2020-10-28 20:00 UTC', '2020-10-28 23:00 UTC',
        '2020-10-29 00:00 UTC', '2020-10-29 07:00 UTC', '2020-10-29 14:00 UTC', '2020-10-29 15:00 UTC', '2020-10-29 19:00 UTC', '2020-10-29 23:00 UTC',
        '2020-10-30 00:00 UTC', '2020-10-30 07:00 UTC', '2020-10-30 14:00 UTC', '2020-10-30 15:00 UTC', '2020-10-30 19:00 UTC', '2020-10-30 20:00 UTC', '2020-10-30 23:00 UTC',
        # additional times for various events
        '2020-10-26 17:00 UTC', '2020-10-27 17:00 UTC', '2020-10-28 16:00 UTC',  '2020-10-28 18:00 UTC', '2020-10-28 21:00 UTC', '2020-10-30 16:00 UTC',
    ]

    # We're reserving one track for ourselves at these times
    reserved_times = [
        '2020-10-26 21:00 UTC', 
        '2020-10-27 18:00 UTC', 
        '2020-10-27 17:00 UTC', 
        # '2020-10-28 22:00 UTC', 
        # '2020-10-29 16:00 UTC', 
        # '2020-10-29 17:00 UTC', 
        # '2020-10-30 17:00 UTC', 
    ] # Unfortunately blocking all of these blows up the scheduler; we'll have to
    # live with 10 parallel tracks at these 4 time slots.

    if not os.path.exists('saved_conf.pickle'):
        conf = load_nmc3()
        # increasing min_shared_times makes scheduling easier, but increases the number of talks that aren't in a session and reduces session coherency
        conf = generate_greedy_schedule(conf, 
                                        blocked_times=blocked_times, 
                                        min_shared_times=1)
        pickle.dump(conf, open('saved_conf.pickle', 'wb'))
    else:
        conf = pickle.load(open('saved_conf.pickle', 'rb'))

    max_tracks = 9
    max_track_vector = reserved_times_to_max_tracks(conf, reserved_times, max_tracks)
    
    # schedule the sessions using ILP
    # if relax==True then it will ignore integer constraints and find a solution usually within a minute or two. This solution usually seems
    # to work pretty well just rounding up, but not 100% (might get some unassigned sessions). You should run relax=True first to check that
    # it is feasible (you'll get an error if not, increase max_tracks), and to give you an idea of the best possible solution you could get in theory.
    # setting relax=False will use the integer constraints which will take a lot longer. A complete run probably around 2-3h, but you can
    # set an upper bound to the computation time and return the best solution found so far. You can also set a bound to how close to the
    # relaxed version you're happy with (e.g. max_mip_gap=0.1 will stop if it finds an integer solution within 10% of the relaxed version).
    relax = True
    if relax:
        model_kwds = None
    else:
        model_kwds = dict( # only relevant if relax=False
            emphasis = 1, # tells the algorithm to concentrate on finding any feasible solution. doesn't make much difference I suspect.
            max_mip_gap = 0.1, # allow a solution within 10% of optimal
            max_seconds = 60*60, # limit computation time to 1h (should be enough to get a good solution)
            )

    # if running this gives an error, it's probably because there's no solution for that number of tracks and you need to increase max_tracks
    
    assign_sessions(conf, 
                    max_track_vector=max_track_vector, 
                    relax=relax)
    pickle.dump(conf, open('saved_conf_assigned.pickle', 'wb'))
    html_schedule_dump(conf, filename=f'schedule_coherent_sessions_dedupped.html')

    slot_unscheduled(conf, blocked_times, max_track_vector)
    html_schedule_dump(conf, filename=f'schedule_coherent_sessions_dedupped_reslotted.html')
    
    pickle.dump(conf, open('saved_conf_assigned_reslotted.pickle', 'wb'))

    # dump to json
    dump_to_json(conf)

    #greedy_assign_sessions(conf)    
    #html_schedule_dump(conf, filename='schedule.html')

    # stats on how many conflicts individual participants have
    # options  = defaultdict(int)
    # for participant in conf.participants:
    #     prefs = set(participant.preferences)
    #     # generate all talks at a given slot
    #     talks = defaultdict(list)
    #     for talk in conf.talks:
    #         if talk in conf.talk_assignment and talk in prefs:
    #             talks[conf.talk_assignment[talk]].append(talk)
    #     if len(talks):
    #         max_options = max(map(len, talks.values()))
    #     else:
    #         max_options = 0
    #     options[max_options] += 1
    # print(options)

    # dists = compute_participant_availability_distributions(conf)
    # for k, v in dists.items():
    #     plt.bar(v.keys(), v.values(), width=1, label=k, alpha=0.5)
    # plt.legend(loc='best')
    # plt.show()
