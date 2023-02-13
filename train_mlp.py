from src.mlp import MLP
from src.data_handler_mixed_sd import DataHandler
import torch
import pickle
import argparse
from torch.nn.functional import sigmoid

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

data_handler = DataHandler("data/Ratings_Warriner_et_al.csv")
data_handler.preprocess(scaling=config.scaling)
data_handler.build_datasets()

with open('public_data/data_handler_mixed.pkl', 'wb') as f:
    pickle.dump(data_handler, f)

layer_dims = [int(x) for x in config.layer_dims.split('|')]
mlp = MLP(layer_dims, do=config.use_dropout, sig=config.use_sigmoid).to('cuda:0')
print("---- MLP parameters ----\n", mlp.parameters, "\n")

criterion = torch.nn.MSELoss(reduction='mean')
# criterion = torch.nn.L1Loss(reduction='mean')
optimizer = torch.optim.Adam(mlp.parameters(), lr=config.lr)
# optimizer = torch.optim.SGD(mlp.parameters(),lr=config.lr)


def threshold_count(x, y, t):
    d = torch.abs(x - y) - t
    d = d.view(-1, 1).squeeze(1)
    d = sigmoid(10000 * d)
    r = torch.sum(d) / len(d)
    return r


valid_loss_min = 1e8
for epoch in range(config.num_epochs):
    # monitor losses
    train_loss = 0
    valid_loss = 0
    l1_loss_txt = 0
    l1_loss_img = 0
    r_txt = 0
    r_img = 0

    ###################
    # train the model #
    ###################
    mlp.train()  # prep model for training
    for data, label, _ in data_handler.txt_train_loader:
        optimizer.zero_grad()
        output = mlp(data, train=True)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    for data, label, _ in data_handler.img_train_loader:
        optimizer.zero_grad()
        output = mlp(data, train=True)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    ######################
    # validate the model #
    ######################
    with torch.no_grad():
        mlp.eval()  # prep model for evaluation
        for data, label, sds in data_handler.txt_test_loader:
            output = mlp(data)
            l1_loss_txt += torch.sum(torch.abs(output - label)).item()
            label = data_handler.scaler_V.unscale(label)
            output = data_handler.scaler_V.unscale(output)
            loss = criterion(output, label)
            valid_loss += loss.item() * data.size(0)
            # l1_loss_txt += torch.sum(torch.abs(output - label) / 8.6).item()
            r_txt += threshold_count(output, label, sds) * data.size(0)

        for data, label, sds in data_handler.img_test_loader:
            output = mlp(data)
            l1_loss_img += torch.sum(torch.abs(output - label)).item()
            label = data_handler.scaler_V.unscale(label)
            output = data_handler.scaler_V.unscale(output)
            loss = criterion(output, label)
            valid_loss += loss.item() * data.size(0)
            # l1_loss_img += torch.sum(torch.abs(output - label) / 8.6).item()
            r_img += threshold_count(output, label, sds) * data.size(0)

        train_loss = train_loss / (
            len(data_handler.txt_train_loader.sampler)
            + len(data_handler.img_train_loader.sampler)
        )
        valid_loss = valid_loss / (
            len(data_handler.txt_test_loader.sampler)
            + len(data_handler.img_test_loader.sampler)
        )
        l1_loss_txt = l1_loss_txt / (3 * len(data_handler.txt_test_loader.sampler))
        l1_loss_img = l1_loss_img / (3 * len(data_handler.img_test_loader.sampler))
        r_txt = r_txt / len(data_handler.txt_test_loader.sampler)
        r_img = r_img / len(data_handler.img_test_loader.sampler)

        if (epoch + 1) % 100 == 0:
            print(
                'Epoch: {} \tTrain Loss: {:.3f} \tVal Loss: {:.3f} \ttxt Loss: {:.3f}, \timg Loss: {:.3f} \ttxt r: {:.4f}, \timg r: {:.4f}'.format(
                    epoch + 1,
                    train_loss,
                    valid_loss,
                    l1_loss_txt,
                    l1_loss_img,
                    r_txt,
                    r_img,
                )
            )

            # save model
            if valid_loss <= valid_loss_min:
                torch.save(mlp.state_dict(), 'data/model_mixed.pt')
                valid_loss_min = valid_loss
