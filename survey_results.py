import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv("data/responses.csv")
col_names = df.columns.values
refs = df[df['Full name:']=='Ideal responder']
df = df.drop(0)

data = {
    "score": [],
    'prompt': [],
    "Mode": []
}
for col_name in col_names:
    if col_name[:8] == 'How well':
        aff = "No AC" if refs[col_name].item() == 1 else "AC"
        prompt = col_name.split('"')[1]
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
        y=0.01,
        xanchor="left",
        x=0.01
    ),
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
fig.add_trace(go.Histogram(histfunc="count", x=matched, name="Match", marker={'color': 'HSL(105,40,40)'}))
fig.add_trace(go.Histogram(histfunc="count", x=unmatched, name="No Match", marker={'color': 'HSL(0,40,50)'}))
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