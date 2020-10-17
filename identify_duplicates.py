from topic_distance import TopicDistance

import collections
import pandas as pd

def main():
    td = TopicDistance()
    td.talks = td.talks.drop('available_dt', axis = 1)
    ntalks = td.talks.shape[0]
    exact_dupes = []
    for i in range(ntalks):
        for j in range(i+1, ntalks):
            sim = td[i, j]
            if sim >= .999:
                exact_dupes.append((i, j))

    # Always keep the abstract with the most votes.
    votes = collections.defaultdict(lambda: 0)
    df_preferences = pd.read_csv('preferences.csv')
    for submission_ids in df_preferences.submission_ids:
        for submission_id in submission_ids.split("'")[1::2]:
            votes[submission_id] += 1
    votes = dict(votes)

    df_talks = pd.read_csv('submissions.csv')
    for dup1, dup2 in exact_dupes:
        if votes[td.talks.loc[dup1].id] < votes[td.talks.loc[dup2].id]:
            df_talks.loc[
                df_talks.id == td.talks.loc[dup1, 'id'], 'submission_status'] = 'Duplicate'
        else:
            df_talks.loc[
                df_talks.id == td.talks.loc[dup2, 'id'], 'submission_status'] = 'Duplicate'
    
    # Withdraw my abstract
    df_talks.loc[df_talks.email == 'patrick.mineault@gmail.com', 'submission_status'] = 'Retracted'
    df_talks = df_talks.drop('Unnamed: 0', axis=1)
    df_talks.to_csv('submissions_dedupped.csv')

if __name__ == "__main__":
    main()