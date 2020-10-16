from schedule_writer import html_schedule_dump
from topic_distance import JaccardDistance, SumDistance, TopicDistance

from collections import defaultdict
import numba
import numpy as np
import pickle

def estimate_goodness(slots, J):
    m = min([len(t) for t in slots])

    goodness = 0
    for j in range(m):
        goodness += (J[slots[0][j], slots[1][j]] + 
                     J[slots[0][j], slots[2][j]] + 
                     J[slots[1][j], slots[2][j]])
    return goodness

def propose_session(slots, J):
    m = min([len(t) for t in slots])

    no_swaps = 0
    while no_swaps < 25:
        # Pick a talk at random
        slot_num = np.random.randint(3)
        slot = slots[slot_num]
        track_num = np.random.randint(m)

        talk1 = slot[track_num]
        if slot_num == 0:
            talka = slots[1][track_num]
            talkb = slots[2][track_num]
        elif slot_num == 1:
            talka = slots[0][track_num]
            talkb = slots[2][track_num]
        elif slot_num == 2:
            talka = slots[0][track_num]
            talkb = slots[1][track_num]

        curr_goodness = -J[talk1, talka] - J[talk1, talkb]

        # What would be the cost of switching out this track?
        S = curr_goodness * np.ones((len(slots), len(slots[0])))
        for i in range(len(slots)):
            for j in range(m):
                if j == track_num:
                    continue
                talk2 = slots[i][j]

                if i == 0:
                    talkc = slots[1][j]
                    talkd = slots[2][j]
                elif i == 1:
                    talkc = slots[0][j]
                    talkd = slots[2][j]
                elif i == 2:
                    talkc = slots[0][j]
                    talkd = slots[1][j]

                # How well would 2 fit in.
                delta_cost = (J[talk2, talka] + J[talk2, talkb]
                            - J[talk2, talkc] - J[talk2, talkd]
                            + J[talk1, talkc] + J[talk1, talkd])

                S[i, j] += delta_cost

        if np.any(S > 0):
            swapi, swapj = np.where(S == S.max())
            old = slots[swapi[0]][swapj[0]]
            slots[slot_num][track_num] = old
            slots[swapi[0]][swapj[0]] = talk1
            no_swaps = 0
        else:
            no_swaps += 1

    return slots

def improve_session(conf, topic_distance=None):
    talks_per_hour = 3
    # generate all talks at a given slot
    talks = defaultdict(list)
    for talk in conf.talks:
        if talk in conf.talk_assignment:
            talks[conf.talk_assignment[talk]].append(talk)

    # now look for shufflable sessions
    J = numba.typed.Dict()
    id2talk = {}
    for _idx, talk in enumerate(conf.talks):
        talk._idx = _idx

    for talk1 in conf.talks:
        id2talk[talk1._idx] = talk1
        for _idx2, talk2 in enumerate(conf.talks):
            J[talk1._idx, talk2._idx] = topic_distance[talk1, talk2]

    conf.similarity_to_successor = {}
    goodness0_cum = 0
    goodness_cum = 0
    ntracks = 0

    for h in range(conf.num_hours):
        slots = []
        for ds in range(talks_per_hour):
            s = h * talks_per_hour + ds
            slots.append(list(talks[s]))

        m = min([len(t) for t in slots])
        if m < 2:# or m>=7:
            continue

        # print(f"Reached {h}/{conf.num_hours}, size {m}")

        for ds in range(3):
            slots[ds] = np.array([t._idx for t in slots[ds]])

        goodness0 = estimate_goodness(slots, J)
        proposed_sessions = []
        goodness = []

        goodness0_cum += goodness0

        # This is in a highly frustrated state, restarting helps a lot
        for i in range(10):
            new_slots = [a.copy() for a in slots]
            new_slots = propose_session(new_slots, J)
            proposed_sessions.append(new_slots)

            # Compute total goodness
            goodness.append(estimate_goodness(new_slots, J))

        goodness = np.array(goodness)
        
        assert goodness.max() > goodness0

        slots = proposed_sessions[goodness.argmax()]

        goodness_cum += goodness.max()
        ntracks += m

        print(f"Coherence before {goodness0:.3}, after {goodness.max():.3}")

        best_session = [[id2talk[t] for t in slot] for slot in slots]

        for i, session in enumerate(zip(*best_session)):
            for n, talk in enumerate(session):
                assert n <= 3
                conf.track_assignment[talk] = i
                conf.talk_assignment[talk] = h * talks_per_hour + n

                #conf.similarity_to_successor[talk] = best_objective
            msg = 'Topic distances: '
            for i in range(len(session)):
                for j in range(i+1, len(session)):
                    talk1 = session[i]
                    talk2 = session[j]
                    d = topic_distance[talk1, talk2]
                    msg += f'({i}-{j}:{d:.2f}) '
            session[0].scheduling_message = msg
            session[1].scheduling_message = ''
            session[2].scheduling_message = ''
    print(f"Average session goodness before: {goodness0_cum / ntracks:.3}")
    print(f"Average session goodness after: {goodness_cum / ntracks:.3}")
    return conf

def main():
    conf = pickle.load(open('saved_conf_sessioned.pickle', 'rb'))
    conf = improve_session(conf, topic_distance=SumDistance(TopicDistance(), JaccardDistance(conf)))

    html_schedule_dump(conf, filename='schedule_sessioned_improved.html')

if __name__ == "__main__":
    main()