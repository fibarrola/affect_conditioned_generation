import torch
import plotly.graph_objects as go
from src.data_handler_bert import DataHandlerBERT


device = "cuda:0" if torch.cuda.is_available() else "cpu"


torch.cuda.empty_cache()

data_handler = DataHandlerBERT("data/Ratings_Warriner_et_al.csv")
prompt_0 = "mountain calm"
prompt_1 = "mountain exciting"

z_0 = data_handler.model.get_learned_conditioning([prompt_0]).to("cpu")
z_1 = data_handler.model.get_learned_conditioning([prompt_1]).to("cpu")

xx = torch.norm((z_0 - z_1).squeeze(0), dim=1)

fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(len(xx))), y=xx.cpu()))
fig.show()

xx = torch.norm((z_0 - z_1).squeeze(0), dim=0)

fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(len(xx))), y=xx.cpu()))
fig.show()

# xx = []

# for channel in range(77):
#     aff_val.requires_grad = False
#     aff_val = aff_val.detach()
#     path = f"data/bert_nets/data_ch_{channel}.pkl"
#     data_handler.load_data(savepath=path)


# mlp = MLP(param_env="mlp.env", h0=768).to(device)
# criterion = torch.nn.MSELoss(reduction='mean')

# z_0 = data_handler.model.get_learned_conditioning([prompt])
# z_0.requires_grad = False
# dim = 0

# for channel in range(77):
#     with open(f'data/bert_nets/data_handler_bert_{channel}.pkl', 'rb') as f:
#         data_handler = pickle.load(f)
#     with torch.no_grad():
#         mlp.load_state_dict(torch.load(f'data/bert_nets/model_{channel}.pt'))

#         out = mlp(data_handler.scaler_Z.scale(z_0[:, channel, :]))[0, dim]

#         xx.append(out.detach().item())

# fig.add_trace(go.Scatter(x=list(range(len(xx))), y=xx, name=prompt))

# fig.show()
