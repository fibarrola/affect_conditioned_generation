import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv("data/v2_responses.csv")
new_names = {col_name:f"{col_name[:2]}_{'most' if k%2==0 else 'least'}" for k, col_name in enumerate(df.columns) if k>=4 and k<=39}
df = df.rename(columns=new_names)
new_names = {col_name:f"{col_name[:2]}" for k, col_name in enumerate(df.columns) if k>=40}
df = df.rename(columns=new_names)

print(df.columns)

# print(df.iloc[0])

# Sorting
sorting_data = []
for k in range(18):
    for choice_of in ["most", "least"]:
        col_name = "{number:02d}_{choice_of:}".format(number=k+1, choice_of=choice_of)
        generator = "VQGAN+CLIP" if (k//3) % 2 ==0 else "StableDifussion"
        affect_dim = "Valence" if (k//6) % 3 ==0 else ("Arousal" if (k//6) % 3 ==1 else "Dominance")
        right_answer = df[col_name].iloc[0]
        for n in range(1, len(df)):

            if df[col_name].iloc[n] == right_answer:
                correctness = "correct"
            else:
                other_choice = "most" if choice_of == "least" else "least"
                correctness = "terrible" if df[col_name].iloc[n] == df["{number:02d}_{choice_of:}".format(number=k+1, choice_of=other_choice)].iloc[0] else "close"
            sorting_data.append({
                "answer": df[col_name].iloc[n],
                "right_answer": right_answer,
                "correctness": correctness,
                "choice_of": choice_of,
                "generator": generator,
                "affect_dim": affect_dim,
            })

sorting_data = pd.DataFrame(sorting_data)
# fig = px.histogram(sorting_data, x="correctness", color="generator", histnorm="percent")
fig = go.Figure()
for generator in ["VQGAN+CLIP", "StableDifussion"]:
    aux = sorting_data[sorting_data["generator"]==generator]
    xx = ["correct", "close", "terrible"]
    yy = [len(aux[aux["correctness"]==correctness])/len(aux) for correctness in xx]
    print(xx, yy)
    fig.add_trace(go.Bar(x = xx, y = yy, name=generator))
fig.show()