import dateutil
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pickle
from collections import defaultdict

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
            self.available = []

    def from_iso(self, available, start_time):
        '''
        Assume available times are either a list of, or a string of iso formatted times, with semicolon separator for strings.
        Assume start_time is either a datetime or iso formatted string.
        '''
        if isinstance(available, str):
            if available=='':
                available = []
            else:
                available = available.split(';')
                available = [a for a in available if a]
        if isinstance(start_time, str):
            start_time = dateutil.parser.isoparse(start_time)
        available = list(map(dateutil.parser.isoparse, available))
        available = [int((t-start_time).total_seconds())//(60*60) for t in available]
        self.available = available
        return self

    def __getitem__(self, i):
        return self.available[i]

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
        try:
            return cls().update_from_dict(s.to_dict()).availability_from_available_dt(start_time)
        except:
            print(s.title)
            raise


class Conference:
    def __init__(self, start_time, end_time):
        if isinstance(start_time, str):
            start_time = dateutil.parser.isoparse(start_time)
        if isinstance(end_time, str):
            end_time = dateutil.parser.isoparse(end_time)
        self.start_time = start_time
        self.end_time = end_time
        self.num_hours = int((end_time-start_time).total_seconds())//(60*60)
        self.participants = []
        self.talks = []
    
    def from_csv(self, submissions='submissions_dedupped.csv', users='users.csv', preferences='preferences.csv', orig_submissions='submissions_with_duplicates.csv'):
        self.talks_df = pd.read_csv(submissions)
        self.talks_df = self.talks_df[self.talks_df.submission_status=="Accepted"]
        self.orig_talks_df = pd.read_csv(orig_submissions)
        self.orig_talks_df = self.orig_talks_df[self.orig_talks_df.submission_status=="Accepted"]
        self.users_df = pd.read_csv(users)
        self.preferences_df = pd.read_csv(preferences)
        # convert into object oriented format
        self.talks = [Talk.from_series(s, self.start_time) for s in self.talks_df.iloc if isinstance(s.available_dt, str)]
        self.id2talk = dict((talk.id, talk) for talk in self.talks)
        self.id2talk.update(dict((talk, talk) for talk in self.talks))
        self.email2talk = dict((talk.email, talk) for talk in self.talks)
        self.id2user = dict((user.id, user) for user in self.users_df.iloc)
        self.id2preference = dict((pref.id, pref.submission_ids) for pref in self.preferences_df.iloc)
        for talk in self.orig_talks_df.iloc:
            if talk.email in self.email2talk:
                self.id2talk[talk.id] = self.email2talk[talk.email]
        missing_submissions = set()
        users_missing_available = set()
        no_corresponding_user = set()
        for pref_row in self.preferences_df.iloc:
            prefs = eval(pref_row.submission_ids)
            prefs = [self.id2talk[pref] for pref in prefs if pref in self.id2talk]
            missing_submissions = missing_submissions.union(set([pref for pref in prefs if pref not in self.id2talk]))
            if pref_row.id not in self.id2user:
                no_corresponding_user.add(pref_row.id)
                # create a fake participant for this
                participant = Participant(available=Availability('', self.start_time), preferences=prefs)
            else:
                user = self.id2user[pref_row.id]
                available_dt = user.available_dt
                if not isinstance(available_dt, str):
                #     participant = Participant(available=Availability(user.available_dt, self.start_time), preferences=prefs)
                #     self.participants.append(participant)
                # else:
                    available_dt = ''
                    users_missing_available.add(user.id)
                participant = Participant(available=Availability(available_dt, self.start_time), preferences=prefs)
            self.participants.append(participant)
        print(f'Missing submission ids: {len(missing_submissions)}')
        print(f'Users with no availability: {len(users_missing_available)}')
        print(f'No corresponding user: {len(no_corresponding_user)}')

    @property
    def num_talks(self):
        return len(self.talks)

    @property
    def num_participants(self):
        return len(self.participants)


def load_nmc3(stats=False):
    # start_time = datetime.datetime(2020, 10, 26, 0, tzinfo=datetime.timezone.utc)
    # end_time = datetime.datetime(2020, 10, 31, 12, tzinfo=datetime.timezone.utc)
    start_time = datetime.datetime(2020, 10, 26, 14, tzinfo=datetime.timezone.utc)
    end_time = datetime.datetime(2020, 10, 30, 23, tzinfo=datetime.timezone.utc)
    conf = Conference(start_time, end_time)
    conf.from_csv()

    if stats:
        df = conf.talks_df
        print(df.talk_format.value_counts())
        print()
        print(df.theme.value_counts())
        print()
        print(f'Number of valid preference voters: {len(conf.participants)}')
        print(f'Number of talks starred: {sum(len(p.preferences) for p in conf.participants)}')
        print(f'Number of times available: {sum(len(p.available) for p in conf.participants)}')
        print()
        print("The following submissions have got too few available times:")
        for s in df.iloc:
            if s.submission_status!="Accepted":
                continue
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
        # 10 most popular talks
        popularity = defaultdict(int)
        for p in conf.participants:
            for t in p.preferences:
                popularity[t] += 1
        popularity = sorted(list(popularity.items()), key=lambda x: x[1], reverse=True)
        print("Top 10 most popular talks:")
        for talk, pop in popularity[:10]:
            print(f' - [{pop} votes] {talk.title}')
            print(f'     {talk.fullname}, {talk.coauthors}')
        # generate participant availability
        plt.figure()
        for i, (src, lab) in enumerate([(conf.participants, 'Participants'), (conf.talks, 'Presenters')]):
            available = np.zeros(24)
            for p in src:
                for t in p.available.available:
                    available[t%24] += 1
            plt.bar(np.arange(24)+0.5*i, available/np.amax(available), width=0.5, label=lab, align='edge')
        plt.legend(loc='best')
        plt.xlabel('Time (UTC)')
        plt.title('Availability')
        plt.yticks([])
        plt.tight_layout()
        plt.savefig('availability.png')
        # preference / theme
        ontheme = 0.0
        numcounted = 0
        mainthemecount = defaultdict(float)
        onthemelist = []
        relative_interest = defaultdict(float)
        relative_interest_count = defaultdict(float)
        for p in conf.participants:
            prefcount = defaultdict(float)
            if len(p.preferences)<=4:
                continue
            for t in p.preferences:
                prefcount[t.theme] += 1
            if prefcount:
                preftotal = sum(prefcount.values())
                for k in prefcount.keys():
                    prefcount[k] /= preftotal
                maintheme = max(prefcount.keys(), key=lambda x: prefcount[x])
                for k in prefcount.keys():
                    relative_interest[maintheme, k] += prefcount[k]#/prefcount[maintheme]
                    relative_interest_count[maintheme, k] += 1
                ontheme += prefcount[maintheme]
                numcounted += 1
                mainthemecount[maintheme] += 1
                onthemelist.append(prefcount[maintheme])
        for k in relative_interest_count.keys():
            relative_interest[k] /= relative_interest_count[k]
        ontheme /= numcounted
        onthemelist = np.array(onthemelist)
        mainthemetotal = sum(mainthemecount.values())
        for k in mainthemecount.keys():
            mainthemecount[k] /= mainthemetotal
        print(f'Percentage of talks within main theme {ontheme*100:.1f}%')
        for k, v in sorted(mainthemecount.items(), key=lambda x: x[1], reverse=True):
            print(f' - {100*v:.1f}% main theme of {k}')

        theme_interest_count = defaultdict(float)
        for p in conf.participants:
            for t in p.preferences:
                theme_interest_count[t.theme] += 1
        ntot = sum(theme_interest_count.values())
        for k in theme_interest_count.keys():
            theme_interest_count[k] /= ntot
        theme_submission_count = defaultdict(float)
        for t in conf.talks:
            theme_submission_count[t.theme] += 1
        ntot = sum(theme_submission_count.values())
        interest_to_submission_ratio = {}
        for k in theme_submission_count.keys():
            theme_submission_count[k] /= ntot
            interest_to_submission_ratio[k] = theme_interest_count[k]/theme_submission_count[k]
        print(f'Interest relative to submissions:')
        for k, v in sorted(interest_to_submission_ratio.items(), key=lambda x: x[1], reverse=True):
            print(f' - {v:.2f} for {k}')

        # plt.figure(figsize=(12, 3))
        plt.figure()
        # plt.subplot(122)
        plt.hist(onthemelist*100, bins=20)
        plt.xlabel('Fraction of interests within main theme')
        plt.xlim(0, 100)
        plt.yticks([])
        # plt.subplot(121)
        # themes = sorted(theme_submission_count.keys())
        # rel_interest = np.full((len(themes), len(themes)), np.nan)
        # for i, theme_i in enumerate(themes):
        #     for j, theme_j in enumerate(themes):
        #         if i==j:
        #             continue
        #         #rel_interest[i, j] = relative_interest[theme_i, theme_j]
        #         rel_interest[i, j] = relative_interest[theme_i, theme_j]/theme_submission_count[theme_j]
        # #plt.imshow(rel_interest, vmin=0, vmax=1, aspect='equal')
        # plt.imshow(rel_interest, aspect='equal')
        # #plt.xticks(np.arange(len(themes)), themes, rotation=20, ha='right')
        # plt.xticks(np.arange(6), 'ABCDEF')
        # plt.yticks(np.arange(len(themes)), themes)
        # plt.xlabel('Secondary themes')
        # plt.ylabel('Main theme')
        # plt.colorbar()
        plt.tight_layout()
        plt.savefig('preference-stats.png')
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
    #conf = load_synthetic('times_and_prefs_2k_500.pickle')
    conf = load_nmc3(stats=True)
    if 1:
        for talk in conf.talks:
            if len(talk.title)>200:
                print(talk.title)
    if 0:
        all_available = np.zeros(24*5+12)
        for sub in conf.talks:
            all_available[sub.available.available] += 1
        plt.bar(range(len(all_available)), all_available, width=1)
        plt.xlabel('Time from 26 Oct, 00:00 UTC (h)')
        plt.title('Available times')
        plt.show()
