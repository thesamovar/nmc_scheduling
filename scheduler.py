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
            for s in range(talks_per_hour*s, talks_per_hour*(s+1)):
                F[talk, s] = 1
                talk_available_slots[talk].append(s)
    if verbose:
        print(f'Finished setting up sparse matrices. {int(time.time()-start_time)}s')
        print(f'  A has {len(A)} entries.')
        print(f'  F has {len(F)} entries.')
        print(f'  I has {len(I)} entries.')

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
    tracks = defaultdict(int)
    remaining_slots = set(range(conf.num_hours*talks_per_hour))
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

    return conf


if __name__=='__main__':
    import pickle
    import matplotlib.pyplot as plt
    start_time = time.time()
    # load and convert synthetic data
    #conf = load_synthetic('times_and_prefs_2k_850.pickle')
    conf = load_nmc3()
    conf = generate_greedy_schedule(conf, max_tracks=6)
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
