import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv("data/Affect_SQ (Responses) - Form responses 1.csv")
df = df.dropna()
# print(df.columns)
print(len(df))
new_names = {col_name:f"{col_name[:2]}_{'most' if k%2==1 else 'least'}" for k, col_name in enumerate(df.columns) if col_name[4:8]=="Pick"}
df = df.rename(columns=new_names)
df = df.rename(columns={df.columns[4]:"computer/phone"})
new_names = {col_name:col_name[:2] for col_name in df.columns if col_name[4:7]=="How"}
df = df.rename(columns=new_names)
# print(df.columns)
print(df)

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
print(sorting_data)
# fig = px.histogram(sorting_data, x="correctness", color="generator", histnorm="percent")
fig = go.Figure()
for generator in ["VQGAN+CLIP", "StableDifussion"]:
    aux = sorting_data[sorting_data["generator"]==generator]
    xx = ["correct", "close", "terrible"]
    yy = [len(aux[aux["correctness"]==correctness])/len(aux) for correctness in xx]
    fig.add_trace(go.Bar(x = xx, y = yy, name=generator))
fig.show()

fig = go.Figure()
for aff_dim in ["Valence", "Arousal", "Dominance"]:
    aux = sorting_data[sorting_data["affect_dim"]==aff_dim]
    xx = ["correct", "close", "terrible"]
    yy = [len(aux[aux["correctness"]==correctness])/len(aux) for correctness in xx]
    fig.add_trace(go.Bar(x = xx, y = yy, name=aff_dim))
fig.show()

assert False

#
# Rating
#
rating_data = []
for k in range(18, 54):
    col_name = "{number:02d}".format(number=k+1)
    generator = "VQGAN+CLIP" if (k//3) % 2 ==0 else "StableDifussion"
    affect_dim = "Valence" if ((k-18)//12) % 3 ==0 else ("Arousal" if ((k-18)//12) % 3 ==1 else "Dominance")
    for n in range(1,len(df)):
        error = abs(df[col_name].iloc[0]-df[col_name].iloc[n])/8
        right_answer = df[col_name].iloc[0]
        rating_data.append({
            "answer": df[col_name].iloc[n],
            "right_answer": right_answer,
            "error": error,
            "generator": generator,
            "affect_dim": affect_dim,
        })

fig = px.box(rating_data, x="error", color="generator")
fig.show()
fig = px.box(rating_data, x="error", color="affect_dim")
fig.show()

order_data = []
for k in range(18, 54):
    col_name = "{number:02d}".format(number=k+1)
    generator = "VQGAN+CLIP" if (k//3) % 2 ==0 else "StableDifussion"
    affect_dim = "Valence" if ((k-18)//12) % 3 ==0 else ("Arousal" if ((k-18)//12) % 3 ==1 else "Dominance")
    if k%3 == 2:
        right_ones = [df["{number:02d}".format(number=m+1)].iloc[0] for m in range(k-2,k+1)]
        for n in range(1,len(df)):
            generator = "VQGAN+CLIP" if (k//3) % 2 ==0 else "StableDifussion"
            affect_dim = "Valence" if ((k-18)//12) % 3 ==0 else ("Arousal" if ((k-18)//12) % 3 ==1 else "Dominance")
            answers = [df["{number:02d}".format(number=m+1)].iloc[n] for m in range(k-2,k+1)]
            right_order = sorted(zip(right_ones, answers))
            order_isright = right_order[0][1] < right_order[1][1] < right_order[2][1]
            order_data.append({
                "order_isright": order_isright,
                "generator": generator,
                "affect_dim": affect_dim,
            })

fig = px.histogram(order_data, x="order_isright", color="generator", histnorm="percent")
fig.show()
fig = px.histogram(order_data, x="order_isright", color="affect_dim", histnorm="percent")
fig.show()
