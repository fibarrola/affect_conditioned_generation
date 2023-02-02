import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv("data/responses.csv")
col_names = df.columns.values
refs = df[df['Full name:']=='Ideal responder']
df = df.drop(0)

# 'Waves hitting the rocks',
# 'The sea at nightfall',
# 'A dark forest',
# 'A windy night',
# 'flaming landscape',
# 'A volcano',
# 'A large rainforest',
# 'butterflys',
# 'Going downriver',
# 'A remote island',
# 'A treasure map',
# 'An old temple',

data = {
    "score": [],
    'prompt': [],
    "Mode": []
}
for col_name in col_names:
    if col_name[:8] == 'How well':
        aff = "No AC" if refs[col_name].item() == 1 else "AC"
        prompt = col_name.split('"')[1]
        if prompt == "Waves hitting the rocks":
            prompt = "Waves hitting <br> the rocks"
        elif prompt == "The sea at nightfall":
            prompt = "The sea at <br> nightfall"
        elif prompt == "A large rainforest":
            prompt = "A large <br> rainforest"
        elif prompt == "A flaming landscape":
            prompt = "Flaming <br> landscape"


        for x in df[col_name]:
            if not np.isnan(x):
                data["score"].append(x)
                data['prompt'].append(prompt)
                data["Mode"].append(aff)

df2 = pd.DataFrame(data=data)
fig = px.box(data_frame=df2, x='prompt', y="score", color="Mode")
fig.update_layout(
    legend=dict(
        yanchor="bottom",
        y=0.0,
        xanchor="left",
        x=0.0
    ),
    yaxis = {
        "title": {
          "standoff": 1
        }
    },
    legend_title=""
)
fig.show()


data = {
    'affect dim': [],
    'match': []
}
matched = []
unmatched = []
for col_name in col_names:
    if col_name[:8] == 'Which of':
        ref = int(refs[col_name])
        if col_name[-2] == 'l':
            aff_dim = "Dominance"
        elif col_name[-2] == 'm':
            aff_dim = "Arousal"
        else:
            aff_dim = "Valence"
        for x in df[col_name]:
            if not x == ' ':
                x = float(x)
                if not np.isnan(x):
                    match = 'Match' if x == ref else "No match"
                    data['match'].append(match)
                    data['affect dim'].append(aff_dim)
                    if x == ref:
                        matched.append(aff_dim)
                    else:
                        unmatched.append(aff_dim)


# To change the order
matched.remove("Valence")
matched.remove("Arousal")
matched.remove("Dominance")
matched = ["Valence", "Arousal", "Dominance"] + matched


fig = go.Figure()
fig.add_trace(go.Histogram(histfunc="count", x=matched, name="Match", marker={'color': "HSL(131,66,70)"}))
fig.add_trace(go.Histogram(histfunc="count", x=unmatched, name="No Match", marker={'color': "HSL(5,67,70)"}))
fig.update_layout(
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    ),
    barmode='group', bargap=0.20,bargroupgap=0.2
)

fig.show()

aff_names = ["Valence", "Arousal", "Dominance"]
matched_counts = [sum([1 for x in matched if x == aff_name]) for aff_name in aff_names]
unmatched_counts = [sum([1 for x in unmatched if x == aff_name]) for aff_name in aff_names]
matched_avgs = [matched_counts[k]/(matched_counts[k]+unmatched_counts[k]) for k in range(3)]
unmatched_avgs = [unmatched_counts[k]/(matched_counts[k]+unmatched_counts[k]) for k in range(3)]

fig = go.Figure()
fig.add_trace(go.Bar(x=aff_names, y=matched_avgs, name="Match", marker={'color': "HSL(131,66,70)"}))
fig.add_trace(go.Bar(x=aff_names, y=unmatched_avgs, name="No Match", marker={'color': "HSL(5,67,70)"}))
fig.update_layout(
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    ),
    barmode='group', bargap=0.20,bargroupgap=0.2
)
fig.update_yaxes(
    tickformat= ',.0%',
    range=[0,1]
)

fig.show()