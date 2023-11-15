import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv("data/Affect_SQ (Responses) - Form responses 1.csv")
df = df.dropna()
# df = df.drop(3)
# df = df.drop(4)
# print(df.columns)
print(len(df))
new_names = {col_name:f"{col_name[:2]}_{'most' if k%2==1 else 'least'}" for k, col_name in enumerate(df.columns) if col_name[4:8]=="Pick"}
df = df.rename(columns=new_names)
df = df.rename(columns={df.columns[4]:"computer/phone"})
new_names = {col_name:f"Q{col_name[:2]}" for col_name in df.columns if col_name[4:7]=="How"}
df = df.rename(columns=new_names)
# print(df.columns)
print(df)

ERRORS = ["Correct", "Small Error", "Large Error"]

#
# Sorting
#
sorting_data = []
for k in range(24):
    for choice_of in ["most", "least"]:
        col_name = "{number:02d}_{choice_of:}".format(number=k+1, choice_of=choice_of)
        generator = "VQGAN+CLIP" if (k//4) % 2 ==0 else "StableDifussion"
        affect_dim = "Valence" if (k//8) % 3 ==0 else ("Arousal" if (k//8) % 3 ==1 else "Dominance")
        right_answer = df[col_name].iloc[0]
        # print(generator, affect_dim, right_answer)
        for n in range(1, len(df)):
            if df[col_name].iloc[n] == right_answer:
                correctness = ERRORS[0]
            else:
                other_choice = "most" if choice_of == "least" else "least"
                correctness = ERRORS[2] if df[col_name].iloc[n] == df["{number:02d}_{choice_of:}".format(number=k+1, choice_of=other_choice)].iloc[0] else ERRORS[1]
            sorting_data.append({
                "answer": df[col_name].iloc[n],
                "right_answer": right_answer,
                "correctness": correctness,
                "choice_of": choice_of,
                "generator": generator,
                "affect_dim": affect_dim,
            })

sorting_data = pd.DataFrame(sorting_data)
print(sorting_data)
# fig = px.histogram(sorting_data, x="correctness", color="generator", histnorm="percent")
fig = go.Figure()
for generator in ["VQGAN+CLIP", "StableDifussion"]:
    aux = sorting_data[sorting_data["generator"]==generator]
    xx = ERRORS
    yy = [len(aux[aux["correctness"]==correctness])/len(aux) for correctness in xx]
    fig.add_trace(go.Bar(x = xx, y = yy, name=generator))
fig.update_layout(
    title="Correct Identification Rates",
    title_x=0.5,
    title_y=0.85,
    yaxis={"tickformat": ',.0%', "range":[0, 1]},
    bargap=0.2,
    bargroupgap=0.2,
    legend={"yanchor":"top", "y":0.99, "xanchor":"right", "x":0.99},
)
fig.show()

fig = go.Figure()
for aff_dim in ["Valence", "Arousal", "Dominance"]:
    aux = sorting_data[sorting_data["affect_dim"]==aff_dim]
    xx = ERRORS
    yy = [len(aux[aux["correctness"]==correctness])/len(aux) for correctness in xx]
    fig.add_trace(go.Bar(x = xx, y = yy, name=aff_dim))
fig.update_layout(
    title={"text":"Correct Identification Rates"},
    title_x=0.5,
    title_y=0.85,
    yaxis={"tickformat": ',.0%', "range":[0, 1]},
    bargap=0.2,
    bargroupgap=0.2,
    legend={"yanchor":"top", "y":0.99, "xanchor":"right", "x":0.99},
)

fig.show()


#
# Quality
#
quality_data = []


# Old survey
df_old = pd.read_csv("data/responses.csv")
col_names = df_old.columns.values
refs = df_old[df_old['Full name:'] == 'Ideal responder']
df_old = df_old.drop(0)
data = {"score": [], 'prompt': [], "Mode": []}
for col_name in col_names:
    if col_name[:8] == 'How well':
        affect_cond = "No affect" if refs[col_name].item() == 1 else "Affect-conditioning" 
        for x in df_old[col_name]:
            if not np.isnan(x):
                quality_data.append({
                    "score": x,
                    "conditioning": affect_cond,
                    "generator": "VQGAN+CLIP"
                })

# New survey

for col_name in df.columns:
    if col_name[0] == "Q":
        affect_cond = "Affect-conditioning" if df[col_name].iloc[0] == 7 else "No affect"
        for n in range(1, len(df)):
            quality_data.append({
                "score": df[col_name].iloc[n],
                "conditioning": affect_cond,
                "generator": "Stable Diffusion"
            })

quality_data = pd.DataFrame(quality_data)
fig = px.box(quality_data, x="generator", y="score", color="conditioning")
fig.update_yaxes(range=[-0.5, 7.5])
fig.update_xaxes(title=None)
fig.update_layout(
    title="Quality Scores",
    title_x=0.5,
    legend={"yanchor":"bottom", "y":0.01, "xanchor":"right", "x":0.99}
)
fig.show()



# rating_data = []
# for k in range(18, 54):
#     col_name = "{number:02d}".format(number=k+1)
#     generator = "VQGAN+CLIP" if (k//3) % 2 ==0 else "StableDifussion"
#     affect_dim = "Valence" if ((k-18)//12) % 3 ==0 else ("Arousal" if ((k-18)//12) % 3 ==1 else "Dominance")
#     for n in range(1,len(df)):
#         error = abs(df[col_name].iloc[0]-df[col_name].iloc[n])/8
#         right_answer = df[col_name].iloc[0]
#         rating_data.append({
#             "answer": df[col_name].iloc[n],
#             "right_answer": right_answer,
#             "error": error,
#             "generator": generator,
#             "affect_dim": affect_dim,
#         })

# fig = px.box(rating_data, x="error", color="generator")
# fig.show()
# fig = px.box(rating_data, x="error", color="affect_dim")
# fig.show()

# order_data = []
# for k in range(18, 54):
#     col_name = "{number:02d}".format(number=k+1)
#     generator = "VQGAN+CLIP" if (k//3) % 2 ==0 else "StableDifussion"
#     affect_dim = "Valence" if ((k-18)//12) % 3 ==0 else ("Arousal" if ((k-18)//12) % 3 ==1 else "Dominance")
#     if k%3 == 2:
#         right_ones = [df["{number:02d}".format(number=m+1)].iloc[0] for m in range(k-2,k+1)]
#         for n in range(1,len(df)):
#             generator = "VQGAN+CLIP" if (k//3) % 2 ==0 else "StableDifussion"
#             affect_dim = "Valence" if ((k-18)//12) % 3 ==0 else ("Arousal" if ((k-18)//12) % 3 ==1 else "Dominance")
#             answers = [df["{number:02d}".format(number=m+1)].iloc[n] for m in range(k-2,k+1)]
#             right_order = sorted(zip(right_ones, answers))
#             order_isright = right_order[0][1] < right_order[1][1] < right_order[2][1]
#             order_data.append({
#                 "order_isright": order_isright,
#                 "generator": generator,
#                 "affect_dim": affect_dim,
#             })

# fig = px.histogram(order_data, x="order_isright", color="generator", histnorm="percent")
# fig.show()
# fig = px.histogram(order_data, x="order_isright", color="affect_dim", histnorm="percent")
# fig.show()
