import dateutil
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pickle

class Availability:
    def __init__(self, available=None, start_time=None):
        if isinstance(available, Availability):
            self.available = available.available
        elif available is not None:
            if isinstance(available, str) or isinstance(available[0], str):
                if start_time is None:
                    raise ValueError("Must provide start_time when setting availability with iso times")
                self.from_iso(available, start_time)
            else:
                self.available = available
        else:
            self.available = set()
        self.available = set(self.available)

    def from_iso(self, available, start_time):
        '''
        Assume available times are either a list of, or a string of iso formatted times, with semicolon separator for strings.
        Assume start_time is either a datetime or iso formatted string.
        '''
        if isinstance(available, str):
            available = available.split(';')
        if isinstance(start_time, str):
            start_time = dateutil.parser.isoparse(start_time)
        available = list(map(dateutil.parser.isoparse, available))
        available = [int((t-start_time).total_seconds())//(60*60) for t in available]
        self.available = set(available)
        return self

    def __len__(self):
        return len(self.available)


class Participant:
    def __init__(self, available=None, preferences=None):
        self.available = Availability(available)
        self.preferences = preferences


class Talk:
    def __init__(self, available=None):
        self.available = Availability(available)
    def update_from_dict(self, d):
        self.__dict__.update(d)
        return self
    def availability_from_available_dt(self, start_time):
        self.available = Availability(self.available_dt, start_time)
        return self
    @classmethod
    def from_csv_row(cls, header, row, start_time):
        return cls().update_from_dict(dict(zip(header, row))).availability_from_available_dt(start_time)
    @classmethod
    def from_series(cls, s, start_time):
        return cls().update_from_dict(s.to_dict()).availability_from_available_dt(start_time)


class Conference:
    def __init__(self, start_time, end_time):
        if isinstance(start_time, str):
            start_time = dateutil.parser.isoparse(start_time)
        if isinstance(end_time, str):
            end_time = dateutil.parser.isoparse(end_time)
        self.start_time = start_time
        self.end_time = end_time
        self.participants = []
        self.talks = []
    
    def talks_from_dataframe(self, df):
        self.talks_df = df
        self.talks = [Talk.from_series(s, self.start_time) for s in df.iloc if isinstance(s.available_dt, str)]

    def talks_from_csv(self, filename):
        df = pd.read_csv(filename)
        self.talks_from_dataframe(df)

    @property
    def num_talks(self):
        return len(self.talks)

    @property
    def num_participants(self):
        return len(self.participants)


def load_nmc3(stats=False):
    start_time = datetime.datetime(2020, 10, 26, 0, tzinfo=datetime.timezone.utc)
    end_time = datetime.datetime(2020, 10, 31, 12, tzinfo=datetime.timezone.utc)
    conf = Conference(start_time, end_time)
    conf.talks_from_csv('submissions.csv')

    if stats:
        df = conf.talks_df
        print(df.talk_format.value_counts())
        print()
        print(df.theme.value_counts())
        print()
        print("The following people have got too few available times:")
        for s in df.iloc:
            if not isinstance(s.available_dt, str) or len(s.available_dt.split(';'))<5:
                print(f"{s.fullname} <{s.email}>;")

        all_available = np.zeros(24*5+12)
        for sub in conf.talks:
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

    return conf


def load_synthetic(filename):
    start_time = datetime.datetime(2020, 10, 26, 0, tzinfo=datetime.timezone.utc)
    end_time = datetime.datetime(2020, 10, 31, 12, tzinfo=datetime.timezone.utc)
    data = pickle.load(open(filename, 'rb'))
    free_times = data['free_times']
    prefs = data['prefs']
    talk_clusters = data['talk_clusters']
    talk_interestingness = data['talk_interestingness']
    num_talks = data['num_talks']
    num_participants = len(prefs)
    talk_type = data['talk_type']

    conf = Conference(start_time, end_time)

    for i in range(num_talks):
        ft = free_times[i]
        talk = Talk()
        talk.available = Availability(ft, start_time)
        talk.talk_format = ['Traditional talk', 'Interactive talk'][talk_type[i]]
        conf.talks.append(talk)

    for i in range(num_participants):
        ft = free_times[i]
        pref = prefs[i]
        participant = Participant(available=ft, preferences=pref)
        conf.participants.append(participant)

    return conf


if __name__=='__main__':
    conf = load_nmc3(stats=True)
    #conf = load_synthetic('times_and_prefs_2k_500.pickle')
    if 0:
        all_available = np.zeros(24*5+12)
        for sub in conf.talks:
            all_available[sub.available.available] += 1
        plt.bar(range(len(all_available)), all_available, width=1)
        plt.xlabel('Time from 26 Oct, 00:00 UTC (h)')
        plt.title('Available times')
        plt.show()
