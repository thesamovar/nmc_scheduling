'''
Automatic scheduler for neuromatch conference
'''

import numpy as np
import mip
from collections import defaultdict
import time
from conference import Conference, Talk, Participant, load_nmc3, load_synthetic

def generate_schedule(conf, strict_interactive_talk_scheduling=False):
    interactive_talks = [i for i, talk in enumerate(conf.talks) if talk.talk_format=="Interactive talk"]
    traditional_talks = [i for i, talk in enumerate(conf.talks) if talk.talk_format=="Traditional talk"]
    num_hours = int((conf.end_time-conf.start_time).total_seconds())//(60*60)
    # Stage 1, greedily assign interactive talk sessions
    remaining = set(interactive_talks)
    while remaining:
        print(len(remaining))
        # compute the set of possible talk time slots for each talk
        session_watch = []
        best_session = None
        best_session_watch = -1
        for slot in range(num_hours-1):
            num_watch = defaultdict(int)
            for talk in remaining:
                # compute number of watch minutes if talk in this slot
                avail = conf.talks[talk].available.available
                if strict_interactive_talk_scheduling:
                    can_sched = slot in avail and slot+1 in avail
                else:
                    can_sched = slot in avail or slot+1 in avail
                if can_sched:
                    for participant in conf.participants:
                        avail = participant.available.available
                        if talk in participant.preferences and slot in avail and slot+1 in avail:
                            num_watch[talk] += 1
            # get the best 7 (TODO: if the best is 3 or smaller, should probably stop there and switch to just doing shorter sessions)
            best_talks = sorted(list(num_watch.items()), key=lambda x: x[1], reverse=True)[:7]
            this_session_watch = sum(n for talk, n in best_talks)
            if this_session_watch>best_session_watch:
                best_session = (slot, best_talks)
                best_session_watch = this_session_watch
            session_watch.append(this_session_watch)

        print(best_session_watch, best_session)
        # remove those session talks from remaining
        slot, best_talks = best_session
        remaining -= set(t for t, n in best_talks)
        if len(best_talks)==0:
            break
        # todo: remove assigned slots from participants free times (or not? not sure)

if __name__=='__main__':
    import pickle
    start_time = time.time()
    # load and convert synthetic data
    conf = load_synthetic('times_and_prefs_2k_500.pickle')
    generate_schedule(conf)