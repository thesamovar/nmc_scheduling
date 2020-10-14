'''
Automatic scheduler for neuromatch conference
'''

import numpy as np
import mip
from collections import defaultdict
import time
from conference import Conference, Talk, Participant, load_nmc3, load_synthetic
import datetime
from schedule_writer import html_schedule_dump, html_participant_dump
import dateutil.parser
from topic_distance import TopicDistance
import itertools
import numba

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


# Greedy scheduler
def generate_greedy_schedule(conf, max_tracks=np.inf, estimated_audience=10_000,
                             blocked_times=None,
                             strict_interactive_talk_scheduling=False, verbose=True):
    start_time = time.time()
    availability_distributions = compute_participant_availability_distributions(conf)
    interactive_talks = [i for i, talk in enumerate(conf.talks) if talk.talk_format=="Interactive talk"]
    traditional_talks = [i for i, talk in enumerate(conf.talks) if talk.talk_format=="Traditional talk"]
    num_hours = conf.num_hours
    talks_per_hour = 3

    mean_hours_free = np.mean([len(p.available) for p in conf.participants])
    mean_num_interested = np.mean([len(p.preferences) for p in conf.participants])

    ideal_hours = 0
    for p in conf.participants:
        ideal_hours += min(len(p.available), len(p.preferences)/talks_per_hour)

    if verbose:
        print("Mean hours free", mean_hours_free)
        print("Max hours per participant", ideal_hours/conf.num_participants)
        print("Max total hours", ideal_hours)
        print("Mean num interested", mean_num_interested)
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
    I = defaultdict(int)
    A = defaultdict(int)
    F = defaultdict(int)
    talk_available_slots = defaultdict(list)
    for p in conf.participants:
        for s in p.available:
            for s in range(talks_per_hour*s, talks_per_hour*(s+1)):
                A[p, s] = 1
        for t in p.preferences:
            if not isinstance(t, Talk):
                t = conf.talks[t]
            I[p, t] = 1
    for t, talk in enumerate(conf.talks):
        for s in talk.available:
            if s in blocked_times:
                continue
            for s in range(talks_per_hour*s, talks_per_hour*(s+1)):
                F[talk, s] = 1
                talk_available_slots[talk].append(s)
    if verbose:
        print(f'Finished setting up sparse matrices. {int(time.time()-start_time)}s')
        print(f'  A has {len(A)} entries.')
        print(f'  F has {len(F)} entries.')
        print(f'  I has {len(I)} entries.')

    # First, compute how popular each talk
    popularity = dict((t, 0) for t in conf.talks)
    for (p, t) in I.keys():
        popularity[t] += 1

    # Sort talks in order of popularity
    popularity = list(popularity.items())
    popularity.sort(key=lambda x: x[1], reverse=True)

    # # Sort talks in increasing order of availability (better for fitting in fixed number of tracks, worse overall)
    # popularity = list(popularity.items())
    # popularity.sort(key=lambda x: len(x[0].available.available))

    # now go through talks and assign them to their best slot, then eliminate
    # availability for participants for that slot
    RA = A.copy()
    hours = 0
    talk_assignment = dict()
    participant_schedule = defaultdict(list)
    tracks = defaultdict(int)
    remaining_slots = set(range(conf.num_hours*talks_per_hour))
    free_participants_by_slot = defaultdict(set)
    for p, s in A.keys():
        free_participants_by_slot[s].add(p)
    participants_assigned_to_talk = defaultdict(set)
    for t, _ in popularity:
        s_choose = None
        for ignore_track_limit in [False, True]:
            audience_total = defaultdict(int)
            audience = defaultdict(list)
            for p, s in RA.keys():
                if (p, t) in I and (t, s) in F and (ignore_track_limit or s in remaining_slots):
                    audience_total[s] += 1
                    audience[s].append(p)
            if audience_total:
                aud_max = max(audience_total.values())
                equally_good_times = [(tracks[s], s) for s, a in audience_total.items() if a==aud_max]
            else:
                equally_good_times = [(tracks[s], s) for s in talk_available_slots[t] if ignore_track_limit or s in remaining_slots]
            if len(equally_good_times)==0:
                continue
            equally_good_times.sort(key = lambda x: x[0])
            _, s_choose = equally_good_times[0]
            break
        if s_choose is None:
            print(f"Cannot schedule talk {t.title}")
            print(f"  Available:", [avail_time for avail_time in t.available.available if avail_time not in blocked_times])
            continue
        s = s_choose
        talk_assignment[t] = s
        tracks[s] += 1
        if tracks[s]==max_tracks:
            remaining_slots.remove(s)
        P = audience[s]
        hours += len(P)/talks_per_hour
        for p in P:
            participant_schedule[p].append((t, s))
            del RA[p, s]
            free_participants_by_slot[s].remove(p)
            participants_assigned_to_talk[t, s].add(p)
    # go through solution and try to even out sessions
    talks_at_slot = defaultdict(list)
    talks_in_hour = defaultdict(int)
    for (t, s) in talk_assignment.items():
        talks_at_slot[s].append(t)
        talks_in_hour[s//talks_per_hour] += 1
    nt_min = {}
    nt_max = {}
    def recompute_nt_min_max():
        for h in range(conf.num_hours):
            nt_min[h] = np.inf
            nt_max[h] = -np.inf
            for ds in range(talks_per_hour):
                s = h*talks_per_hour+ds
                nt_min[h] = min(len(talks_at_slot[s]), nt_min[h])
                nt_max[h] = max(len(talks_at_slot[s]), nt_max[h])
    recompute_nt_min_max()
    did_moves = 1
    while did_moves:
        did_moves = 0
        for target_session_min_size in range(talks_per_hour-1, 0, -1): # move widows to fill up sessions of size 2 first, then 1
            for hour in range(conf.num_hours):
            # for src_s in range(conf.num_hours*talks_per_hour):
            #     hour = src_s//talks_per_hour
            #     widowed = talks_in_hour[hour]%talks_per_hour==1 and len(talks_at_slot[src_s])>nt_min[hour]
            #     if len(talks_at_slot[src_s])>=max_tracks or widowed:
                if talks_in_hour[hour]%talks_per_hour==1: # widowed talk
                    # identify the slot with one more than the others
                    if len(talks_at_slot[hour*talks_per_hour])>nt_min[hour]:
                        ds = 0
                    elif len(talks_at_slot[hour*talks_per_hour+1])>nt_min[hour]:
                        ds = 1
                    else:
                        ds = 2
                    # identify slots we could move a talk to
                    options = [s for s in range(conf.num_hours*talks_per_hour) if len(talks_at_slot[s])<nt_max[s//talks_per_hour] and talks_in_hour[s//talks_per_hour]%talks_per_hour>=target_session_min_size and s//talks_per_hour!=hour]
                    if len(options)==0:
                        break # we've done all we can
                    # consider which talk to move
                    src_s = hour*talks_per_hour+ds
                    delta_audience = {}
                    s_best = {}
                    aud = {}
                    for talk in talks_at_slot[src_s]:
                        # and what is the best place we could move it to?
                        potential_audience = {}
                        for s in options:
                            if F[talk, s]:
                                potential_audience[s] = [p for p in free_participants_by_slot[s] if I[p, talk]]
                        if len(potential_audience)==0:
                            continue
                        s_best[talk], aud[talk] = max(potential_audience.items(), key=lambda x: len(x[1]))
                        delta_audience[talk] = len(aud[talk])-len(participants_assigned_to_talk[talk, src_s])
                    if len(delta_audience)==0:
                        continue # we can't reschedule any of these
                    talk, _ = max(delta_audience.items(), key=lambda x: x[1])
                    s_best = s_best[talk]
                    aud = aud[talk]
                    #print(f'Moving talk {talk.title} from {src_s//talks_per_hour} to {s_best//talks_per_hour} with delta {delta_audience[talk]}')
                    did_moves += 1
                    # remove this talk 
                    talks_at_slot[src_s].remove(talk)
                    talks_in_hour[src_s//talks_per_hour] -= 1
                    for p in participants_assigned_to_talk[talk, src_s]:
                        free_participants_by_slot[src_s].add(p)
                        participant_schedule[p].remove((talk, src_s))
                    del participants_assigned_to_talk[talk, src_s]
                    # add it back
                    talks_at_slot[s_best].append(talk)
                    talks_in_hour[s_best//talks_per_hour] += 1
                    talk_assignment[talk] = s_best
                    participants_assigned_to_talk[talk, s_best] = aud
                    for p in aud:
                        free_participants_by_slot[s_best].remove(p)
                        participant_schedule[p].append((talk, s_best))
                    recompute_nt_min_max()

    # create sessions by assigning to tracks
    audience_size = defaultdict(float)
    for p, sched in participant_schedule.items():
        for (t, s) in sched:
            audience_size[t] += 1
    # initial track assignment by popularity
    track_assignment = {}
    for s in range(conf.num_hours*talks_per_hour):
        talks_with_pop = sorted([(talk, audience_size[talk]) for talk in talks_at_slot[s]], key=lambda x: x[1], reverse=True)
        for track, (talk, _) in enumerate(talks_with_pop):
            track_assignment[talk] = track
    
    # Some stats on the found solution
    scaling = estimated_audience/conf.num_participants
    if verbose:
        print(f"Found solution.")
        print(f"  Total hours: {hours*scaling:.1f} (ideal would be {ideal_hours*scaling:.1f})")
        print(f"  Talks assigned: {len(talk_assignment)} of {conf.num_talks}.")
        tracks = defaultdict(int)
        for (t, s) in talk_assignment.items():
            tracks[s] += 1
        max_tracks = max(tracks.values())
        print(f"  Maximum number of tracks is {max_tracks}")
        audience_size_all = defaultdict(float)
        audience_size_traditional = defaultdict(float)
        audience_size_interactive = defaultdict(float)
        for p, sched in participant_schedule.items():
            for (t, s) in sched:
                audience_size_all[t] += 1
                if t.talk_format=="Traditional talk":
                    audience_size_traditional[t] += 1
                else:
                    audience_size_interactive[t] += 1
        audience_size_all = np.array(list(audience_size_all.values()))
        audience_size_traditional = np.array(list(audience_size_traditional.values()))
        audience_size_interactive = np.array(list(audience_size_interactive.values()))
        print(f"  Audience size (all) min {audience_size_all.min()*scaling:.1f}, max {audience_size_all.max()*scaling:.1f}, mean {audience_size_all.mean()*scaling:.1f}, std {audience_size_all.std()*scaling:.1f}")
        print(f"  Audience size (traditional) min {audience_size_traditional.min()*scaling:.1f}, max {audience_size_traditional.max()*scaling:.1f}, mean {audience_size_traditional.mean()*scaling:.1f}, std {audience_size_traditional.std()*scaling:.1f}")
        print(f"  Audience size (interactive) min {audience_size_interactive.min()*scaling:.1f}, max {audience_size_interactive.max()*scaling:.1f}, mean {audience_size_interactive.mean()*scaling:.1f}, std {audience_size_interactive.std()*scaling:.1f}")        

    # verify found solution
    for t, s in talk_assignment.items():
        if (t, s) not in F:
            raise ValueError((t, s))
    for p, sched in participant_schedule.items():
        for (t, s) in sched:
            if (t, s) not in F or (p, t) not in I or (p, s) not in A:
                raise ValueError(p, t, s, sched)

    conf.talk_assignment = talk_assignment
    conf.participant_schedule = participant_schedule
    conf.track_assignment = track_assignment

    return conf

def sessions_by_similarity_pairs(conf, topic_distance=None):
    talks_per_hour = 3
    # generate all talks at a given slot
    talks = defaultdict(list)
    for talk in conf.talks:
        if talk in conf.talk_assignment:
            talks[conf.talk_assignment[talk]].append(talk)
    # now look for shufflable sessions
    J = {}
    conf.similarity_to_successor = {}
    for h in range(conf.num_hours):
        slots = {}
        for ds in range(talks_per_hour):
            s = h*talks_per_hour+ds
            slots[ds] = list(talks[s])

        if min([len(t) for t in slots.values()])<2:
            continue

        # compute Jaccard similarity between talks
        participants_interested_in = defaultdict(set)
        for p in conf.participants:
            for t in p.preferences:
                participants_interested_in[t].add(p)
        # setup and solve model
        model = mip.Model()
        edge = {}
        for ds0 in range(talks_per_hour-1):
            for i1, t1 in enumerate(slots[ds0]):
                for i2, t2 in enumerate(slots[ds0+1]):
                    if (t1, t2) not in J:
                        if topic_distance is None:
                            J[t1, t2] = len(participants_interested_in[t1] & participants_interested_in[t2])/len(participants_interested_in[t1] | participants_interested_in[t2])
                        else:
                            J[t1, t2] = topic_distance[t1, t2]
                    edge[ds0, t1, t2] = model.add_var(f'edge({ds0}, {i1}, {i2})', var_type=mip.BINARY)
                model += mip.xsum(edge[ds0, t1, t2] for t2 in slots[ds0+1])<=1
            for t2 in slots[ds0+1]:
                model += mip.xsum(edge[ds0, t1, t2] for t1 in slots[ds0])<=1
        model.objective = mip.maximize(mip.xsum(edge[ds0, t1, t2]*(1+J[t1, t2]) for ds0, t1, t2 in edge.keys()))
        model.verbose = 0
        opt_status_relax = model.optimize(relax=False)
        # print(opt_status_relax)
        # for (ds0, t1, t2), e in edge.items():
        #     if e.x:
        #         print(f'{ds0}: [{t1.title}]-[{t2.title}]')
        # reallocate tracks using the model
        succ = {}
        for (ds0, t1, t2), e in edge.items():
            if e.x:
                succ[t1] = t2
                conf.similarity_to_successor[t1] = J[t1, t2]
                #conf.track_assignment[t2] = conf.track_assignment[t1]
        unassigned = set().union(*map(set, slots.values()))
        start_track = defaultdict(int)
        for ds0 in range(talks_per_hour):
            for talk in slots[ds0]:
                if talk in unassigned:
                    conf.track_assignment[talk] = start_track[ds0]
                    unassigned.remove(talk)
                    start_track[ds0] += 1
                if talk in succ:
                    conf.track_assignment[succ[talk]] = conf.track_assignment[talk]
                    unassigned.remove(succ[talk])
    return conf


def sessions_by_similarity_complete(conf, topic_distance=None):
    talks_per_hour = 3
    # generate all talks at a given slot
    talks = defaultdict(list)
    for talk in conf.talks:
        if talk in conf.talk_assignment:
            talks[conf.talk_assignment[talk]].append(talk)
    # now look for shufflable sessions
    J = {}
    if talks_per_hour==3:
        J = numba.typed.Dict()
        id2talk = {}
        for talk1 in conf.talks:
            id2talk[id(talk1)] = talk1
            for talk2 in conf.talks:
                J[id(talk1), id(talk2)] = topic_distance[talk1, talk2]
    conf.similarity_to_successor = {}
    for h in range(conf.num_hours):
        slots = {}
        for ds in range(talks_per_hour):
            s = h*talks_per_hour+ds
            slots[ds] = list(talks[s])

        m = min([len(t) for t in slots.values()])
        if m<2:# or m>=7:
            continue

        print(f"Reached {h}/{conf.num_hours}, size {m}")

        if talks_per_hour==3:
            for ds in range(3):
                slots[ds] = np.array([id(t) for t in slots[ds]])
            @numba.jit(nopython=True)
            def perm(arr):
                n = len(arr)
                c = np.zeros(n+1, dtype=np.int32)
                yield arr.copy()
                i = 0
                while i<n:
                    if c[i]<i:
                        if i%2==0:
                            arr[0], arr[i] = arr[i], arr[0]
                        else:
                            arr[c[i]], arr[i] = arr[i], arr[c[i]]
                        yield arr.copy()
                        c[i] += 1
                        i = 0
                    else:
                        c[i] = 0
                        i += 1
            @numba.jit(nopython=True)
            def partial_objective(slot0, slot1, J):
                objective = 0.0
                for i in range(len(slot0)):
                    for j in range(i+1, len(slot1)):
                        objective += J[slot0[i], slot1[j]]
                return objective
            @numba.jit(nopython=True)
            def find_best_session(slot0, slots1, slots2, J):
                best_objective = -1.0
                for slot1 in perm(slots1):
                    for slot2 in perm(slots2):
                        objective = 0.0
                        objective += partial_objective(slot0, slot1, J)
                        objective += partial_objective(slot1, slot2, J)
                        objective += partial_objective(slot0, slot2, J)
                        if objective>best_objective:
                            best_objective = objective
                            best_session = (slot0, slot1, slot2)
                return best_objective, best_session
            best_objective, best_session = find_best_session(slots[0], slots[1], slots[2], J)
            best_session = [[id2talk[t] for t in slot] for slot in best_session]
        else:
            slot_iterators = [[slots[0]]]+[itertools.permutations(slots[ds]) for ds in range(1, talks_per_hour)]
            session_options = itertools.product(*slot_iterators)
            best_objective = -1
            best_session = None
            for session_option in session_options:
                objective = 0.0
                for session in zip(*session_option):
                    for i in range(len(session)):
                        talk_i = session[i]
                        for j in range(i+1, len(session)):
                            talk_j = session[j]
                            objective += topic_distance[talk_i, talk_j]
                if objective>best_objective:
                    best_objective = objective
                    best_session = session_option
        for i, session in enumerate(zip(*best_session)):
            for talk in session:
                conf.track_assignment[talk] = i
                #conf.similarity_to_successor[talk] = best_objective
    return conf

if __name__=='__main__':
    import os, pickle
    import matplotlib.pyplot as plt
    start_time = time.time()
    if not os.path.exists('saved_conf.pickle'):
        #conf = load_synthetic('times_and_prefs_2k_850.pickle')
        conf = load_nmc3()
        blocked_times = [
            '2020-10-26 00:00 UTC', '2020-10-26 07:00 UTC', '2020-10-26 08:00 UTC', '2020-10-26 14:00 UTC', '2020-10-26 15:00 UTC', '2020-10-26 19:00 UTC', '2020-10-26 20:00 UTC', '2020-10-26 23:00 UTC',
            '2020-10-27 00:00 UTC', '2020-10-27 07:00 UTC', '2020-10-27 08:00 UTC', '2020-10-27 14:00 UTC', '2020-10-27 15:00 UTC', '2020-10-27 19:00 UTC', '2020-10-27 20:00 UTC', '2020-10-27 23:00 UTC',
            '2020-10-28 00:00 UTC', '2020-10-28 07:00 UTC', '2020-10-28 08:00 UTC', '2020-10-28 14:00 UTC', '2020-10-28 15:00 UTC', '2020-10-28 19:00 UTC', '2020-10-28 20:00 UTC', '2020-10-28 23:00 UTC',
            '2020-10-29 00:00 UTC', '2020-10-29 07:00 UTC', '2020-10-29 08:00 UTC', '2020-10-29 14:00 UTC', '2020-10-29 15:00 UTC', '2020-10-29 19:00 UTC', '2020-10-29 20:00 UTC', '2020-10-29 23:00 UTC',
            '2020-10-30 00:00 UTC', '2020-10-30 07:00 UTC', '2020-10-30 08:00 UTC', '2020-10-30 14:00 UTC', '2020-10-30 15:00 UTC', '2020-10-30 19:00 UTC', '2020-10-30 20:00 UTC', '2020-10-30 23:00 UTC',
            ]
        conf = generate_greedy_schedule(conf, max_tracks=6, blocked_times=blocked_times)
        pickle.dump(conf, open('saved_conf.pickle', 'wb'))
    else:
        conf = pickle.load(open('saved_conf.pickle', 'rb'))
    #conf = sessions_by_similarity_pairs(conf, topic_distance=TopicDistance())
    conf = sessions_by_similarity_complete(conf, topic_distance=TopicDistance())
    html_schedule_dump(conf)

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
