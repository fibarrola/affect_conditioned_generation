import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

data = [
    ["A close-up snake", "V", 0.45, "Img. ground truth"],
    ["A close-up snake", "A", 0.71, "Img. ground truth"],
    ["A close-up snake", "D", 0.43, "Img. ground truth"],
    ["A close-up snake", "V", 0.45, "Pred. from image"],
    ["A close-up snake", "A", 0.62, "Pred. from image"],
    ["A close-up snake", "D", 0.51, "Pred. from image"],
    ["A close-up snake", "V", 0.41, "Pred. from prompt"],
    ["A close-up snake", "A", 0.64, "Pred. from prompt"],
    ["A close-up snake", "D", 0.46, "Pred. from prompt"],
    ["An antelope drinking water", "V", 0.86, "Img. ground truth"],
    ["An antelope drinking water", "V", 0.71, "Pred. from image"],
    ["An antelope drinking water", "V", 0.70, "Pred. from prompt"],
    ["An antelope drinking water", "A", 0.41, "Img. ground truth"],
    ["An antelope drinking water", "A", 0.41, "Pred. from image"],
    ["An antelope drinking water", "A", 0.44, "Pred. from prompt"],
    ["An antelope drinking water", "D", 0.79, "Img. ground truth"],
    ["An antelope drinking water", "D", 0.67, "Pred. from image"],
    ["An antelope drinking water", "D", 0.65, "Pred. from prompt"],
    ["A car crash", "V", 0.45, "Img. ground truth"],
    ["A car crash", "V", 0.39, "Pred. from image"],
    ["A car crash", "V", 0.34, "Pred. from prompt"],
    ["A car crash", "A", 0.75, "Img. ground truth"],
    ["A car crash", "A", 0.74, "Pred. from image"],
    ["A car crash", "A", 0.69, "Pred. from prompt"],
    ["A car crash", "D", 0.40, "Img. ground truth"],
    ["A car crash", "D", 0.43, "Pred. from image"],
    ["A car crash", "D", 0.42, "Pred. from prompt"],
    ["A smiling woman sitting on the beach", "V", 0.78, "Img. ground truth"],
    ["A smiling woman sitting on the beach", "V", 0.77, "Pred. from image"],
    ["A smiling woman sitting on the beach", "V", 0.87, "Pred. from prompt"],
    ["A smiling woman sitting on the beach", "A", 0.53, "Img. ground truth"],
    ["A smiling woman sitting on the beach", "A", 0.49, "Pred. from image"],
    ["A smiling woman sitting on the beach", "A", 0.56, "Pred. from prompt"],
    ["A smiling woman sitting on the beach", "D", 0.65, "Img. ground truth"],
    ["A smiling woman sitting on the beach", "D", 0.72, "Pred. from image"],
    ["A smiling woman sitting on the beach", "D", 0.72, "Pred. from prompt"],
]

df = pd.DataFrame(data=data, columns=["Image Description", "dim", "score", "type"])


# fig = px.bar(df, x="Image Description", color="dim", y="score")
# fig.update_layout(barmode='group')
# fig.show()

# for prompt in ["A close-up snake", "An antelope drinking water", "A car crash", "A smiling woman sitting on the beach"]:
#     fil_df = df[df["Image Description"]==prompt]
#     fig = px.bar(fil_df, x="dim", color="type", y="score")
#     fig.update_layout(barmode='group', title=f".    {prompt}", bargap=0.20,bargroupgap=0.2)
#     fig.update_yaxes(range=[0,1])
#     fig.show()


# colors = ["RGB(163,172,247)", "RGB(234,159,151)", "RGB(151,234,159)"]
colors = [
    "HSL(233,86,70)",
    "HSL(5,67,70)",
    "HSL(131,66,70)",
]
descriptions = [
    "A close-up snake",
    "An antelope drinking water",
    "A car crash",
    "A smiling woman sitting on the beach",
]
fig = make_subplots(rows=1, cols=4, subplot_titles=descriptions, shared_yaxes=True,)

for p, prompt in enumerate(descriptions):
    fil_df = df[df["Image Description"] == prompt]
    for n, name in enumerate(
        ["Img. ground truth", "Pred. from image", "Pred. from prompt"]
    ):
        fil_df2 = fil_df[fil_df["type"] == name]
        print(fil_df2)
        fig.add_trace(
            go.Bar(
                x=fil_df2["dim"],
                y=fil_df2["score"],
                name=name,
                marker={'color': colors[n]},
                showlegend=p == 0,
            ),
            row=1,
            col=p + 1,
        )
        fig.update_layout(barmode='group', bargap=0.20, bargroupgap=0.2)
        fig.update_yaxes(range=[0, 1])

fig.show()
