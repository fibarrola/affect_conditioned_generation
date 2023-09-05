from src.mlp import MLP
from src.data_handler_bert import DataHandlerBERT
import torch
import pickle
import argparse


parser = argparse.ArgumentParser(description='Training Parameters')
parser.add_argument("--num_epochs", type=int, default=2500)
parser.add_argument(
    "--scaling",
    type=str,
    default="uniform",
    help="scaling for input and output data. Can be 'uniform', 'whiten', 'normalize' or 'none'",
)
parser.add_argument("--lr", type=float, help="learning rate", default=0.00007)
parser.add_argument(
    "--layer_dims", type=str, help="layer dimensions. Separate with |", default="64|32"
)
parser.add_argument(
    "--use_dropout", type=bool, help="Use dropout for training?", default=True
)
parser.add_argument(
    "--use_sigmoid",
    type=bool,
    help="Use sigmoid at the end of last layer?",
    default=False,
)
config = parser.parse_args()
layer_dims = [int(x) for x in config.layer_dims.split('|')]
criterion = torch.nn.MSELoss(reduction='mean')
data_handler = DataHandlerBERT("data/Ratings_Warriner_et_al.csv")

channel_losses = [0 for x in range(77)]
loss_hist = [[] for x in range(77)]

for channel in range(1, 77):
    print(f"----- Training channel {channel} -----")
    path = f"data/bert_nets/data_ch_{channel}.pkl"
    data_handler.preprocess(savepath=path,scaling=config.scaling, channel=channel)
    data_handler.load_data(savepath=path)
    data_handler.build_datasets()

    mlp = MLP(layer_dims, do=config.use_dropout, sig=config.use_sigmoid, h0=768).to(
        'cuda:0'
    )
    optimizer = torch.optim.Adam(mlp.parameters(), lr=config.lr)

    valid_loss_min = 1e8

    for epoch in range(config.num_epochs):
        # monitor losses
        train_loss = 0
        valid_loss = 0
        l1_loss_txt = 0

        mlp.train()  # prep model for training
        for data, label, _ in data_handler.train_loader:
            optimizer.zero_grad()
            output = mlp(data, train=True)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        with torch.no_grad():
            mlp.eval()  # prep model for evaluation
            for data, label, sds in data_handler.test_loader:
                output = mlp(data)
                label = data_handler.scaler_V.unscale(label)
                output = data_handler.scaler_V.unscale(output)
                loss = criterion(output, label)
                valid_loss += loss.item() * data.size(0)
                l1_loss_txt += torch.sum(torch.abs(output - label) / 8.6).item()

            train_loss = train_loss / len(data_handler.train_loader.sampler)
            valid_loss = valid_loss / len(data_handler.test_loader.sampler)
            l1_loss_txt = l1_loss_txt / (3 * len(data_handler.test_loader.sampler))

            if (epoch + 1) % 100 == 0:
                print(
                    'Epoch: {} \tTrain Loss: {:.3f} \tVal Loss: {:.3f} \ttxt Loss: {:.3f}'.format(
                        epoch + 1, train_loss, valid_loss, l1_loss_txt
                    )
                )

            # save model
            if valid_loss <= valid_loss_min:
                torch.save(mlp.state_dict(), f'data/bert_nets/model_ch_{channel}.pt')
                valid_loss_min = valid_loss
                loss_hist[channel].append(valid_loss)


with open(f'data/bert_nets/loss_hist.pkl', 'wb') as f:
    pickle.dump(loss_hist, f)