import numpy as np
from collections import defaultdict
import time
from conference import Conference, Talk, Participant, load_nmc3, load_synthetic
import datetime

def html_schedule_dump(conf, estimated_audience=10_000, filename='schedule.html'):
    audience_scaling = estimated_audience/conf.num_participants
    talks_per_hour = 3
    # generate all talks at a given slot
    talks = defaultdict(list)
    for talk in conf.talks:
        if talk in conf.talk_assignment:
            talks[conf.talk_assignment[talk]].append(talk)
    max_tracks = max(map(len, talks.values()))
    # popularity
    audience_size = defaultdict(float)
    if len(audience_size):
        for p, sched in conf.participant_schedule.items():
            for (t, s) in sched:
                audience_size[t] += 1
    else:
        available_at = defaultdict(list)
        interested_in = defaultdict(set)
        for p in conf.participants:
            for h in p.available.available:
                available_at[h].append(p)
            for t in p.preferences:
                interested_in[t].add(p)
        for s in talks.keys():
            for p in available_at[s//3]:
                for t in talks[s]:
                    if p in interested_in[t]:
                        audience_size[t] += 1
                        break
    if audience_size:
        max_audience_size = max(audience_size.values())
    else:
        max_audience_size = 1
    # dump solution into an html file
    current_day = 'This is not a day'
    current_hour = -1
    html_rows = ''
    for s in range(conf.num_hours*talks_per_hour):
        T = set(talks[s])
        conflict = 0
        for p in conf.participants:
            c = max(sum(1 for t in p.preferences if t in T)-1, 0)
            conflict += c
        conflict *= audience_scaling
        h = s//talks_per_hour
        m = 15*(s%talks_per_hour)
        t_o = conf.start_time+datetime.timedelta(seconds=60*60*h+m*60)
        t = t_o.strftime('%a %H:%M')
        if t_o.strftime('%a')!=current_day:
            current_day = t_o.strftime('%a')
            html_rows += f'<td class="day" colspan="{max_tracks+3}">{current_day}</td>'
        row = f'<td>{t}</td><td>{int(conflict)} missed</td>'
        #curtalks = talks[s]
        #curtalks = sorted([(audience_size[talk], talk) for talk in curtalks], reverse=True, key=lambda x:x[0])
        #curtalks = sorted([(audience_size[talk], talk) for talk in curtalks], key=lambda x: conf.track_assignment[x[1]])
        # fill up curtalks and do a quick check that there are no repeats
        curtalks = {}
        for talk in talks[s]:
            track = conf.track_assignment[talk]
            if (track, talk) in curtalks:
                print('ERROR! Multiple assignment detected')
            curtalks[track] = talk
        #curtalks = dict((conf.track_assignment[talk], talk) for talk in talks[s])
        total_viewers = int(sum(audience_size[talk] for talk in talks[s])*audience_scaling)
        row += f'<td>{total_viewers} viewers</td>'
        for track in range(max_tracks):
            #if track<len(curtalks):
            if track in curtalks:
                talk = curtalks[track]
                size = audience_size[talk]
                title = talk.title
                if len(title)>100:
                    title = f'<span title="{title}">{title[:100]}...</span>'
                if isinstance(talk.coauthors, str):
                    coauth = talk.coauthors
                else:
                    coauth = talk.fullname
                authors = f'<span title="{coauth}">{talk.fullname}</span>'
                estim = size*audience_scaling
                if hasattr(conf, 'similarity_to_successor'):
                    sim = conf.similarity_to_successor.get(talk, None)
                else:
                    sim = None
                if sim is None:
                    sim = ''
                else:
                    sim = f'<br/>Similarity to next: {sim:.3f}'
                if hasattr(talk, 'scheduling_message'):
                    msg = '<br/>'+talk.scheduling_message
                else:
                    msg = ''
                c = f'''
                    <div class="talk">
                        <span style="font-size: 80%">{talk.talk_format}</span><br/>
                        <b>{title}</b><br/>
                        <i>{authors}</i><br/>
                        <span style="font-size: 80%">{int(estim)} viewers (estim.){sim}{msg}</span>
                    </div>
                    '''
                pop = size/max_audience_size
                bgcol = f'hsl({180-180*pop:.0f}, 50%, 75%)'
                rowclass = 'class="talk_cell"'
            else:
                c = ''
                bgcol = '#ffffff'
                rowclass = ''
            
            row += f'''
                <td {rowclass} style="background-color: {bgcol};">
                    {c}
                </td>
                '''
        if m==30:
            html_rows += f'<tr class="lastinsession">{row}</tr>'
        else:
            html_rows += f'<tr>{row}</tr>'
    track_numbers = ''.join(f'<th>Track {i+1}</th>' for i in range(max_tracks))
    css = '''
    table { border-collapse: collapse; }
    th, td {
        text-align: left;
        padding: 8px;
    }
    th { background-color: #eeeeee; }
    .talk {
        width: 10em;
    }
    .talk_cell {
        border-left: 1px solid black;
        border-right: 1px solid black;
    }
    tr { border-bottom: 1px dashed #888888; }
    th, .lastinsession, .day { border-bottom: 1px solid black;}
    .day { background-color: #eeccee; }
    '''
    html = f'''
    <html>
        <head>
            <title>Schedule</title>
            <style>
                {css}
            </style>
        </head>
        <body>
            <table>
                <tr>
                    <th>Time (UTC)</th>
                    <th>Conflict (missed talks)</th>
                    <th>Viewers (estimated)</th>
                    {track_numbers}
                </tr>
                {html_rows}
            </table>
        </body>
    </html>
    '''
    open(filename, 'wb').write(html.encode('UTF-8'))

#html_schedule_dump(conf)

#%%
def html_participant_dump(conf, participant, fname):
    talks_per_hour = 3
    prefs = set(participant.preferences)
    # generate all talks at a given slot
    talks = defaultdict(list)
    for talk in conf.talks:
        if talk in conf.talk_assignment and talk in prefs:
            talks[conf.talk_assignment[talk]].append(talk)
    max_tracks = max(map(len, talks.values()))
    # popularity
    audience_size = defaultdict(float)
    for p, sched in conf.participant_schedule.items():
        for (t, s) in sched:
            audience_size[t] += 1
    max_audience_size = max(audience_size.values())
    # dump solution into an html file
    html_rows = ''
    for s in range(conf.num_hours*talks_per_hour):
        T = set(talks[s])
        if len(T)==0:
            continue
        h = s//talks_per_hour
        m = 15*(s%talks_per_hour)
        t = conf.start_time+datetime.timedelta(seconds=60*60*h+m*60)
        t = t.strftime('%a %H:%M')
        row = f'<td>{t}</td>'
        curtalks = talks[s]
        curtalks = sorted([(audience_size[talk], talk) for talk in curtalks], reverse=False, key=lambda x:x[0])
        for track in range(max_tracks):
            if track<len(curtalks):
                size, talk = curtalks[track]
                title = talk.title
                if len(title)>100:
                    title = f'<span title="{title}">{title[:100]}...</span>'
                if isinstance(talk.coauthors, str):
                    coauth = talk.coauthors
                else:
                    coauth = talk.fullname
                authors = f'<span title="{coauth}">{talk.fullname}</span>'
                c = f'''
                    <div class="talk">
                        <b>{title}</b><br/>
                        <i>{authors}</i><br/>
                    </div>
                    '''
                bgcol = '#eeeeee'
            else:
                c = ''
                bgcol = '#ffffff'
            
            row += f'''
                <td style="background-color: {bgcol};">
                    {c}
                </td>
                '''
        html_rows += f'<tr>{row}</tr>'
    track_numbers = ''.join(f'<th>Option {i+1}</th>' for i in range(max_tracks))
    css = '''
    th, td {
        text-align: left;
        padding: 8px;
    }
    th { background-color: #eeeeee; }
    .talk {
        width: 10em;
    }
    tr { border-bottom: 1px solid black;}
    '''
    html = f'''
    <html>
        <head>
            <title>Schedule</title>
            <style>
                {css}
            </style>
        </head>
        <body>
            <table>
                <tr>
                    <th>Time (UTC)</th>
                    {track_numbers}
                </tr>
                {html_rows}
            </table>
        </body>
    </html>
    '''
    open(f'participant_schedules/{fname}.html', 'wb').write(html.encode('UTF-8'))

# for i, participant in enumerate(conf.participants[:30]):
#     html_participant_dump(conf, participant, f'test{i}')
