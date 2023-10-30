from src.mlp import MLP
from src.data_handler_bert_v2 import DataHandlerBERT
import torch
import pickle
import os
from dotenv import load_dotenv

param_env = "bert.env"

load_dotenv(param_env)

criterion = torch.nn.MSELoss(reduction='mean')
data_handler = DataHandlerBERT()

channel_losses = [0 for x in range(77)]
loss_hist = [[] for x in range(77)]

for channel in [3]: #range(77):
    print(f"----- Training channel {channel} -----")
    path = f"{os.environ.get('MODEL_PATH')}/data_ch_{channel}.pkl"
    data_handler.preprocess(
        csv_path=os.environ.get("WORD_DATA_PATH"),
        sdconfig=os.environ.get("SD_CONFIG"),
        sdmodel=os.environ.get("SD_MODEL"),
        savepath=path,
        v_scaling=os.environ.get("V_SCALING"),
    )
    # data_handler.load_data(savepath=path)
    data_handler.build_datasets()

    mlp = MLP(
        h0=int(os.environ.get('IMG_SIZE')),
        use_dropout=os.environ.get('USE_DROPOUT'),
        use_sigmoid=os.environ.get('USE_SIGMOID'),
    ).to('cuda:0')
    optimizer = torch.optim.Adam(mlp.parameters(), lr=float(os.environ.get("LR")))

    valid_loss_min = 1e8

    for epoch in range(int(os.environ.get("TRAINING_EPOCHS"))):
        # monitor losses
        train_loss = 0
        valid_loss = 0
        l1_loss_txt = 0

        mlp.train()  # prep model for training
        for data, label in data_handler.train_loader:
            optimizer.zero_grad()
            output = mlp(data, train=True)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        with torch.no_grad():
            mlp.eval()  # prep model for evaluation
            for data, label in data_handler.test_loader:
                # print(label)
                # assert False
                output = mlp(data)
                loss = criterion(output, label)
                valid_loss += loss.item() * data.size(0)
                l1_loss_txt += torch.sum(torch.abs(output - label)).item()

            train_loss = train_loss / len(data_handler.train_loader.sampler)
            valid_loss = valid_loss / len(data_handler.test_loader.sampler)
            l1_loss_txt = l1_loss_txt / (3 * len(data_handler.test_loader.sampler))

            if (epoch) % 50 == 0:
                print(label)
                print(torch.mean(torch.abs(output - label), dim=0))
                assert False
                print(
                    'Epoch: {} \tTrain Loss: {:.3f} \tVal Loss: {:.3f} \ttxt Loss: {:.3f}'.format(
                        epoch + 1, train_loss, valid_loss, l1_loss_txt
                    )
                )

            # save model
            if valid_loss <= valid_loss_min:
                torch.save(
                    mlp.state_dict(),
                    f"{os.environ.get('MODEL_PATH')}/model_ch_{channel}.pt",
                )
                valid_loss_min = valid_loss
            
            loss_hist[channel].append(valid_loss)


with open(f"data/loss_hist_bert8.pkl", "wb") as f:
    pickle.dump(loss_hist, f)
