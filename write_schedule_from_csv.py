from collections import defaultdict
import time
from conference import Conference, Talk, Participant, load_nmc3, load_synthetic
import datetime
import pandas
import numpy as np
import dateutil


def write_static_html_schedule(filename='submissions-final.csv'):
    talks_df = pandas.read_csv(filename)
    talk_at = defaultdict(list)
    all_times = set()
    for talk in talks_df.iloc:
        if talk.submission_status=="Accepted" and isinstance(talk.track, str):
            if talk.track=="stage":
                track = 0
            else:
                track = int(talk.track[5:])
            # if talk.track[6:]=="":
            #     print(talk.title, talk.fullname)
            # track = int(talk.track[6:])
            #t = dateutil.parser.parse(talk.starttime+' UTC')
            t = dateutil.parser.parse(talk.starttime)
            talk_at[t, track] = talk
            all_times.add(t)
    table_rows = []
    all_times = sorted(list(all_times))
    num_tracks = 1+max(track for t, track in talk_at.keys())
    start_time = datetime.datetime(2020, 10, 26, 13, tzinfo=datetime.timezone.utc)
    end_time = datetime.datetime(2020, 10, 31, 00, tzinfo=datetime.timezone.utc)
    t = start_time
    #_talk_end = dict((str(talk.fullname)+str(talk.title), dateutil.parser.parse(talk.endtime+' UTC')) for talk in talks_df.iloc)
    _talk_end = dict((str(talk.fullname)+str(talk.title), dateutil.parser.parse(talk.endtime)) for talk in talks_df.iloc)
    talk_end = lambda talk: _talk_end[str(talk.fullname)+str(talk.title)]
    def next_time(t):
        dt = t-start_time
        h = dt.total_seconds()//(60*60)
        m = dt.total_seconds()//60-h*60
        firstinhour = m==0
        if m==30:
            idt = datetime.timedelta(seconds=60*30)
            hourchange = True
        else:
            idt = datetime.timedelta(seconds=60*15)
            hourchange = False
        tnext = t+idt
        return tnext, hourchange, firstinhour
    def find_row_span(t, talk):
        # t is current time, set the rowspan to go to the start of the next hour or the end of the talk
        t, hourchange, firstinhour = next_time(t)
        rowspan = 1
        while t<talk_end(talk) and not hourchange:
            rowspan += 1
            t, hourchange, firstinhour = next_time(t)
        return rowspan
    ongoing = {}
    while t<end_time:
        tnext, hourchange, firstinhour = next_time(t)
        talk_at[t].sort(key=lambda x: x[0])
        cell = f'''
        <td class="time">
            <script type="text/JavaScript">LT("{t.strftime('%Y-%m-%d %H:%M')}");</script>
        </td>
        '''
        row = [cell]
        has_contributed = any(talk_at[t, track].talk_format in ('Interactive talk', 'Traditional talk') for track in range(num_tracks) if (t, track) in talk_at)
        spanned_tracks = set()
        for track in range(num_tracks):
            if track in ongoing:
                if t>=talk_end(ongoing[track]):
                    del ongoing[track]
            if (t, track) not in talk_at:
                if track not in ongoing:
                    row.append('<td></td>')
                elif firstinhour: # renew
                    span = f'rowspan="{find_row_span(t, ongoing[track])}"'
                    cell = f'''<td class="talk_cell {ongoing[track].talk_format.replace(' ', '_')}" {span}>... continues ...</td>'''
                    row.append(cell)
            else:
                talk = talk_at[t, track]
                ongoing[track] = talk
                cell = [getattr(talk, v) for v in ['title', 'fullname'] if isinstance(getattr(talk, v), str)]
                if len(cell)==2 and cell[0]==cell[1]:
                    cell = [cell[0]]
                cell[0] = f'<b>{cell[0]}</b>'
                cell = '<br/>'.join(cell)
                details = []
                details_summary = []
                if isinstance(talk.coauthors, str):
                    details.append(f'<div class="coauthors">{talk.coauthors}</div>')
                    details_summary.append('Coauthors')
                if isinstance(talk.abstract, str):
                    abstract = talk.abstract.replace('\n', '<br/>')
                    details.append(f'<div class="abstract">{abstract}</div>')
                    details_summary.append('Abstract')
                details = '<br/>'.join(details)
                details_summary = ', '.join(details_summary)
                if details:
                    details = f'''
                    <div>&nbsp;</div>
                    <details>
                        <summary>{details_summary}</summary>
                        <div>&nbsp;</div>
                        {details}
                    </details>
                    '''
                if talk.talk_format not in ('Interactive talk', 'Traditional talk'):
                    span = f'rowspan="{find_row_span(t, talk)}"'
                    spanned_tracks.add(track)
                else:
                    span = ''
                cell = f'''
                <td class="talk_cell {talk.talk_format.replace(' ', '_')}" {span}>
                    <i>{talk.talk_format}</i><br/>
                    {cell}
                    {details}
                </td>
                '''
                row.append(cell)
        if tnext is None or tnext.hour!=t.hour:
            rowclass = 'class="lastinsession"'
        else:
            rowclass = ''
        row = '\n'.join(row)
        row = f'''
        <script type="text/JavaScript">
            oldday = NewDay(oldday, "{t.strftime('%Y-%m-%d %H:%M')}");
        </script>
        <tr {rowclass}>
            {row}
        </tr>
        '''
        table_rows.append(row)
        t = tnext
    table = '\n'.join(table_rows)
    css = '''
    * {
        font-family: "Trebuchet MS", Helvetica, sans-serif;
    }
    table {
        border-collapse: collapse;
        table-layout: fixed;
    }
    th, td {
        text-align: left;
        padding: 8px;
        vertical-align: top;
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
    .day {
        //background-color: #eeccee;
        font-size: 300%;
        font-weight: bold;
    }
    .Interactive_talk { background-color: #f2f2fa; }
    .Traditional_talk { background-color: #f2faf2; }
    .Special_Event { background-color: #ffeeee; }
    .Keynote_Event { background-color: #ffeeff; }
    .coauthors {
        font-size: 80%;
        font-style: italic;
    }
    .abstract {
        font-size: 80%;
    }
    summary {
        font-size: 80%;
    }
    '''
    js = r'''
	function LT(t) {
        var m = moment.utc(t).tz(moment.tz.guess());
		document.write(m.format("LT"));
	}
    function NewDay(old, t) {
        var m = moment.utc(t).tz(moment.tz.guess());
        newday = m.format("dddd MMMM D");
        if(newday!=old) {
            document.write('<tr><td class="day" colspan="COLSPAN">'+newday+'</td></tr>');
        }
        return newday;
    }
    var oldday = "NonsenseValue";
    function openAll() {
        var elems = document.getElementsByTagName("details");
        document.getElementById("btnExpandHideAllDetails").innerHTML = "Hide details";
        document.getElementById("btnExpandHideAllDetails").setAttribute( "onClick", "javascript: closeAll();");

        for (var i = 0; i < elems.length; i++){
            elems[i].setAttribute("open", "true");
        }
    }

    function closeAll() {	
        var elems = document.getElementsByTagName("details");
        document.getElementById("btnExpandHideAllDetails").setAttribute( "onClick", "javascript: openAll();" );
        document.getElementById("btnExpandHideAllDetails").innerHTML = "Expand all details (do this to enable searching abstracts/coauthors in page)";	
        
        for (var i = 0; i < elems.length; i++){
            elems[i].removeAttribute("open");
        }
    }    
    '''.replace("COLSPAN", f'{num_tracks+1}')
    html = f'''
    <!doctype html>

    <html lang="en">
    <head>
        <meta charset="utf-8">    
        <script type="text/JavaScript" src="https://MomentJS.com/downloads/moment.js"></script>
        <script type="text/JavaScript" src="https://momentjs.com/downloads/moment-timezone-with-data.min.js"></script>
        <title>Neuromatch 3.0 provisional schedule</title>
        <style>
            {css}
        </style>
        <script type="text/JavaScript">
            {js}
        </script>
    </head>
    <body>
        <h1>Neuromatch 3.0 provisional schedule</h1>
        <p>
            Times are given in the following time zone: 
            <script type="text/JavaScript">
                document.write(moment.tz.guess());
            </script>.
        </p>
        <button id="btnExpandHideAllDetails" onclick="openAll()">Expand all details (do this to enable searching abstracts/coauthors in page)</button>
        <p>&nbsp;</p>
        <table>
            {table}
        </table>
    </body>
    </html>
    '''
    open('index.html', 'wb').write(html.encode('UTF-8'))

if __name__=='__main__':
    write_static_html_schedule()
