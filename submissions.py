import csv
import pandas as pd
import dateutil
import datetime

CONFERENCE_START_TIME = datetime.datetime(2020, 10, 26, 0, tzinfo=datetime.timezone.utc)

class Submission:
    def __init__(self, submission_dict):
        for key, value in submission_dict.items():
            setattr(self, key, value)
        if isinstance(self.available_dt, str):
            available = self.available_dt.split(';')
            available = list(map(dateutil.parser.isoparse, available))
            t0 = CONFERENCE_START_TIME
            available = [int((t-t0).total_seconds())//(60*60) for t in available]
            self.available = available
    @classmethod
    def from_csv_row(cls, header, row):
        return Submission(dict(zip(header, row)))
    @classmethod
    def from_series(cls, s):
        return Submission(s.to_dict())

class Submissions:
    def __init__(self, df):
        self.df = df
        self.submission = [Submission.from_series(s) for s in df.iloc]
    def __len__(self):
        return len(self.submission)
    def __getitem__(self, i):
        return self.submission[i]
    @classmethod
    def from_csv(cls, filename='submissions.csv'):
        df = pd.read_csv(filename)
        return cls(df)

if __name__=='__main__':

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import numpy as np

    submissions = Submissions.from_csv()

    print(submissions.df.talk_format.value_counts())
    print(submissions.df.theme.value_counts())
    
    all_available = np.zeros(24*5+12)
    for sub in submissions:
        if hasattr(sub, 'available'):
            all_available[sub.available] += 1
    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    plt.bar(range(len(all_available)), all_available, width=1)
    plt.xlabel('Time from 26 Oct, 00:00 UTC (h)')
    plt.title('Available times')
    plt.subplot(212)
    df.theme.value_counts().plot(kind='barh')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig('submission_stats.png')