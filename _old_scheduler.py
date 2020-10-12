# def generate_schedule(conf, max_tracks=5, strict_interactive_talk_scheduling=False):
#     start_time = time.time()
#     availability_distributions = compute_participant_availability_distributions(conf)
#     interactive_talks = [i for i, talk in enumerate(conf.talks) if talk.talk_format=="Interactive talk"]
#     traditional_talks = [i for i, talk in enumerate(conf.talks) if talk.talk_format=="Traditional talk"]
#     num_hours = conf.num_hours
#     num_slots = (num_hours-1)*max_tracks
#     # Generate input matrices
#     I = np.zeros((conf.num_participants, conf.num_talks), dtype=int)
#     F = np.zeros((conf.num_talks, num_slots), dtype=int)
#     for t in interactive_talks:
#         talk = conf.talks[t]
#         avail = talk.available.available
#         for h in range(num_hours-1):
#             if strict_interactive_talk_scheduling:
#                 can_sched = h in avail and h+1 in avail
#             else:
#                 can_sched = h in avail or h+1 in avail
#             if can_sched:
#                 F[t, h*max_tracks:(h+1)*max_tracks] = 1
#     for p, participant in enumerate(conf.participants):
#         for t in participant.preferences:
#             if t in interactive_talks:
#                 I[p, t] = 1
#     print(f'Finished setting up sparse matrices. {int(time.time()-start_time)}s')
#     print(f'  F has {F.sum()} non-zero entries, shape {F.shape}, size {F.size}.')
#     print(f'  I has {I.sum()} non-zero entries, shape {I.shape}, size {I.size}.')

#     # create decision variables
#     S = cvxpy.Variable((conf.num_talks, num_slots), boolean=True)
#     V = cvxpy.Variable((conf.num_participants, num_slots), boolean=True)
#     print(f"Finished setting up decision variables. {int(time.time()-start_time)}s")
#     print(f"  S: {S.size}, V: {V.size}, total {S.size+V.size}")

#     # add constraints

#     constraints = []
#     constraints.append(cvxpy.sum(S, axis=1)<=1)
#     constraints.append(S<=F)
#     # todo: the rest

#     print(f"Finished setting up constraints. {int(time.time()-start_time)}s")

#     print(I.shape, S.shape, V.shape)

#     objective = cvxpy.sum(cvxpy.multiply(I@S, V))

#     prob = cvxpy.Problem(cvxpy.Maximize(objective), constraints)
#     prob.solve()

#     # for t in interactive_talks:
#     #     model += mip.xsum(S[t, h, k] for h in range(num_hours) if (t, h) in F for k in range(max_tracks))<=1
#     #     nconstraints += 1
#     # for p in range(conf.num_participants):
#     #     for h in range(num_hours-1):
#     #         model += mip.xsum(V[p, h2, k] for k in range(max_tracks) for h2 in range(h, min(h+2, num_hours-1)))<=1
#     #         nconstraints += 1
#     print(f"Finished setting up constraints. {int(time.time()-start_time)}s")
#     print(f"  Number of constraints is {len(constraints)}")
#     return
#     # set objective
#     model.objective = mip.maximize(mip.xsum(V[p,h,k]*S[t,h,k] for (p, t) in I.keys() for h in range(num_hours-1) for k in range(max_tracks)))
#     # solve relaxed first
#     opt_status_relax = model.optimize(relax=True)
#     print(opt_status_relax)
#     relaxed_opt = model.objective_value


# def generate_schedule(conf, max_tracks=5, strict_interactive_talk_scheduling=False):
#     start_time = time.time()
#     availability_distributions = compute_participant_availability_distributions(conf)
#     interactive_talks = [i for i, talk in enumerate(conf.talks) if talk.talk_format=="Interactive talk"]
#     traditional_talks = [i for i, talk in enumerate(conf.talks) if talk.talk_format=="Traditional talk"]
#     num_hours = conf.num_hours
#     # Generate input matrices
#     I = defaultdict(int)
#     F = defaultdict(int)
#     for t in interactive_talks:
#         talk = conf.talks[t]
#         avail = talk.available.available
#         for h in range(num_hours-1):
#             if strict_interactive_talk_scheduling:
#                 can_sched = h in avail and h+1 in avail
#             else:
#                 can_sched = h in avail or h+1 in avail
#             if can_sched:
#                 F[t, h] = 1
#     for p, participant in enumerate(conf.participants):
#         for t in participant.preferences:
#             if t in interactive_talks:
#                 I[p, t] = 1
#     print(f'Finished setting up sparse matrices. {int(time.time()-start_time)}s')
#     print(f'  F has {len(F)} entries.')
#     print(f'  I has {len(I)} entries.')
#     # create model
#     model = mip.Model()
#     # create decision variables
#     S = {}
#     for (t, h) in F.keys():
#         for k in range(max_tracks):
#             S[t, h, k] = model.add_var(f'S({t},{h},{k})', var_type=mip.BINARY)
#     V = {}
#     for p in range(conf.num_participants):
#         for h in range(num_hours-1):
#             for k in range(max_tracks):
#                 V[p, h, k] = model.add_var(f'V({p},{h},{k})', var_type=mip.BINARY)
#     print(f"Finished setting up decision variables. {int(time.time()-start_time)}s")
#     print(f"  S: {len(S)}, V: {len(V)}, total {len(S)+len(V)}")
#     # add constraints
#     nconstraints = 0
#     for t in interactive_talks:
#         model += mip.xsum(S[t, h, k] for h in range(num_hours) if (t, h) in F for k in range(max_tracks))<=1
#         nconstraints += 1
#     for p in range(conf.num_participants):
#         for h in range(num_hours-1):
#             model += mip.xsum(V[p, h2, k] for k in range(max_tracks) for h2 in range(h, min(h+2, num_hours-1)))<=1
#             nconstraints += 1
#     print(f"Finished setting up constraints. {int(time.time()-start_time)}s")
#     print(f"  Number of constraints is {nconstraints}")
#     # set objective
#     model.objective = mip.maximize(mip.xsum(V[p,h,k]*S[t,h,k] for (p, t) in I.keys() for h in range(num_hours-1) for k in range(max_tracks)))
#     # solve relaxed first
#     opt_status_relax = model.optimize(relax=True)
#     print(opt_status_relax)
#     relaxed_opt = model.objective_value



# def generate_schedule(conf, strict_interactive_talk_scheduling=False):
#     availability_distributions = compute_participant_availability_distributions(conf)
#     interactive_talks = [i for i, talk in enumerate(conf.talks) if talk.talk_format=="Interactive talk"]
#     traditional_talks = [i for i, talk in enumerate(conf.talks) if talk.talk_format=="Traditional talk"]
#     num_hours = conf.num_hours

#     # available_participants = defaultdict(set)
#     # for participant in conf.participants:
#     #     for avail in participant.available:
#     #         for talk in participant.preferences:
#     #             available_participants[talk, avail].add(participant)

#     # Stage 1, greedily assign interactive talk sessions
#     remaining = set(interactive_talks)
#     while remaining:
#         print(len(remaining))
#         # greedy algorithm: based on the remaining talks, and which participants are already committed
#         # at which times, what would be the next best session. To answer, we ask for each possible
#         # session time, which would be the best? For a given session time, we ask which collection
#         # of talks would maximise the number of people attending?

#         # compute the set of possible talk time slots for each talk
#         session_watch = []
#         best_session = None
#         best_session_watch = -1
#         for slot in range(num_hours-1):
#             num_watch = defaultdict(int)
#             for talk in remaining:
#                 # compute number of watch minutes if talk in this slot
#                 avail = conf.talks[talk].available.available
#                 if strict_interactive_talk_scheduling:
#                     can_sched = slot in avail and slot+1 in avail
#                 else:
#                     can_sched = slot in avail or slot+1 in avail
#                 if can_sched:
#                     for participant in conf.participants:
#                         avail = participant.available.available
#                         if talk in participant.preferences and slot in avail and slot+1 in avail:
#                             num_watch[talk] += 1
#             # get the best 7 (TODO: if the best is 3 or smaller, should probably stop there and switch to just doing shorter sessions)
#             best_talks = sorted(list(num_watch.items()), key=lambda x: x[1], reverse=True)[:7]
#             this_session_watch = sum(n for talk, n in best_talks)
#             if this_session_watch>best_session_watch:
#                 best_session = (slot, best_talks)
#                 best_session_watch = this_session_watch
#             session_watch.append(this_session_watch)

#         print(best_session_watch, best_session)
#         # remove those session talks from remaining
#         slot, best_talks = best_session
#         remaining -= set(t for t, n in best_talks)
#         if len(best_talks)==0:
#             break
#         # todo: remove assigned slots from participants free times (or not? not sure)
