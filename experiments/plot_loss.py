import pickle
import numpy as np
import plotly.graph_objects as go

with open(f"data/loss_hist_8.pkl", "rb") as f:
    loss_hist = pickle.load(f)

loss_hist = np.array(loss_hist)
loss_hist = loss_hist[:700, :]
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=list(range(loss_hist.shape[0])), y=loss_hist[:, 0], name="text MAE (val)")
)
fig.add_trace(
    go.Scatter(x=list(range(loss_hist.shape[0])), y=loss_hist[:, 1], name="image MAE (val)")
)
fig.update_layout(
    title="Training with CLIP latents",
    title_x=0.5,
    xaxis_title="Epoch",
    legend={
        "yanchor":"top",
        "y":0.99,
        "xanchor":"right",
        "x":0.99
    }
)
fig.show()  

# with open(f"data/bert_nets_wide/loss_hist.pkl", "rb") as f:
#     loss_hist = pickle.load(f)

# fig = go.Figure()
# for losses in loss_hist:
#     fig.add_trace(
#         go.Scatter(x=list(range(len(losses))), y=losses)
#     )
# fig.show()