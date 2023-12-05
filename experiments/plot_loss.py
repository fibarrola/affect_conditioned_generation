import pickle
import numpy as np
import plotly.graph_objects as go

with open("data/loss_hist_8_mse.pkl", "rb") as f:
    loss_hist = pickle.load(f)

loss_hist = np.array(loss_hist)
loss_hist = loss_hist[:700, :]
print("text MAE", loss_hist[-1, 0])
print("img MAE", loss_hist[-1, 1])
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=list(range(loss_hist.shape[0])), y=loss_hist[:, 0], name="text MAE (val)"
    )
)
fig.add_trace(
    go.Scatter(
        x=list(range(loss_hist.shape[0])), y=loss_hist[:, 1], name="image MAE (val)"
    )
)
fig.update_layout(
    title="Training with CLIP latents",
    title_x=0.5,
    title_y=0.85,
    xaxis_title="Epoch",
    legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
)
fig.show()

with open("data/loss_hist_bert8_mse.pkl", "rb") as f:
    loss_hist = pickle.load(f)

fig = go.Figure()
loss_hist = 10 * np.array(loss_hist)
means = np.mean(loss_hist, axis=0)
sds = np.std(loss_hist, axis=0)
print(means[-1], sds[-1])
xx = list(range(len(means)))
y_low = list(means - sds)
y_upp = list(means + sds)
fig.add_trace(
    go.Scatter(
        x=xx + xx[::-1],
        y=y_upp + y_low[::-1],
        fillcolor="rgba(99,110,250,0.5)",
        fill='tozerox',
        line={"color": "rgba(99,110,250,0)"},
        showlegend=False,
    )
)
fig.add_trace(
    go.Scatter(
        x=xx,
        y=means,
        line={"color": "rgba(99,110,250,1)"},
        name="Average text MAE (val)",
    )
)
fig.update_layout(
    title="Training with BERT latents",
    title_x=0.5,
    title_y=0.85,
    xaxis_title="Epoch",
    legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
)
fig.show()
