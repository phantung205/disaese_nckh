import argparse
from configs import config_csv
import torch
from torch import nn
from src_csv.neron import dataset
from src_csv import preprocessing
import shutil
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from src_csv.neron.neural_network import DiabetesNN
from torch.optim import AdamW
from sklearn.metrics import classification_report,recall_score,f1_score,roc_auc_score
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import  ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib


def get_args():
    parser = argparse.ArgumentParser(description="train nn ")
    parser.add_argument("--batch_size","-b",type=int,default=config_csv.batch_size)
    parser.add_argument("--epochs","-e",type=int,default=config_csv.epochs)
    parser.add_argument("--learning_rate","-l",type=float,default=config_csv.learning_rate)
    parser.add_argument("--weight_decay", "-w", type=float, default=config_csv.weight_decay,help="number weight decay")
    parser.add_argument("--trained_models", "-t", type=str, default=config_csv.dir_csv_model, help="model path ")
    parser.add_argument("--logging","-lg", type=str, default=config_csv.path_tensorboard, help="tensorboard path")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="checkpoint")

    args = parser.parse_args()
    return args



def train(args,x_train, x_test, y_train, y_test):
    # chuyển sang GPU
    device =  "cuda" if torch.cuda.is_available() else "cpu"


    # dataloader
    train_dataloader = dataset.create_dataloader(x_train,y_train)
    test_dataloader = dataset.create_dataloader(x_test,y_test,shuffle=False)

    # xóa bỏ tensorboard cũ
    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)

    # khởi tạo tensorboad
    writer = SummaryWriter(args.logging)

    #khởi tạo model
    input_dim = x_train.shape[1]
    model = DiabetesNN(input_dim=input_dim).to(device)

    # khởi tạo hàm loss , giúp imbalance data
    num_positive = (y_train == 1).sum()
    num_negative = (y_train == 0).sum()
    base_pos_weight = 5
    # hệ số điều chỉnh
    # alpha = 0.4
    # pos_weight = base_pos_weight * alpha
    pos_weight = torch.tensor(
        [base_pos_weight],
        dtype=torch.float32,
        device=device
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    # khởi tạo hàm optimizer
    optimizer = AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    # load model checkpoint cux
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 0
        best_acc = 0

    num_iteration = len(train_dataloader)

    for epoch in range(start_epoch,args.epochs):
        model.train()
        progress_bar = tqdm(train_dataloader,colour="cyan")
        for iter,(samples,labels) in enumerate(progress_bar):
            samples = samples.to(device)
            labels = labels.to(device)

            # cho vào model dự đoán
            outputs = model(samples)

            # tính loss
            loss_value = criterion(outputs,labels)

            writer.add_scalar("train/loss",loss_value.item(), epoch*num_iteration+iter)
            progress_bar.set_description("Epoch{}/{}  , iteration {}/{} , loss {:.3f}".format(epoch + 1, args.epochs, iter + 1, num_iteration,loss_value.item()))

            # backward
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0

        all_predictions = []
        all_labels = []
        progress_bar = tqdm(test_dataloader,colour="red")
        for iter, (samples,labels) in enumerate(progress_bar):
            samples = samples.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                predictions = model(samples)
                loss = criterion(
                    predictions,
                    labels
                )
                val_loss += loss.item()

                # probability
                probability = torch.sigmoid(predictions)

                # prediction
                prediction = (probability >= 0.45).float()

                all_predictions.extend(prediction.cpu().numpy().reshape(-1).tolist())

                all_labels.extend(labels.cpu().numpy().reshape(-1).tolist())

        scheduler.step()

        recall = recall_score(all_labels, all_predictions,zero_division=0)
        f1 = f1_score(all_labels, all_predictions,zero_division=0)
        roc_auc = roc_auc_score(all_labels, all_predictions)

        print("Epoch : {}, f1 : {} ".format(epoch + 1, f1))
        writer.add_scalar("val/Accuracy", f1, epoch)

        print(classification_report(all_labels,all_predictions,
                target_names=["No Diabetes","Diabetes"]
            )
        )

        # save model
        checkpoint = {
            "epoch": epoch + 1,
            "best_acc": f1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, "{}/last_nn.pt".format(args.trained_models))
        # save best model ,learning rate ,epochs
        if f1 > best_acc:
            best_acc = f1
            checkpoint = {
                "epoch": epoch + 1,
                "best_acc": best_acc,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(checkpoint, "{}/best_nn.pt".format(args.trained_models))


            # lưu model kết quả đánh giá lại
            if not os.path.isdir(config_csv.dir_evaluate):
                os.makedirs(config_csv.dir_evaluate)
            path_result = os.path.join(config_csv.dir_evaluate, f"train_report_neron.txt")
            with open(path_result, "w") as f:
                f.write(f"Model: neron")
                f.write(classification_report(all_labels,all_predictions,target_names=["No Diabetes","Diabetes"]))
                f.write(f"\nROC-AUC: {roc_auc:.4f}\n")
                f.write(f"\nRecall: {recall:.4f}\n")


    writer.close()

if __name__ == '__main__':
    args = get_args()

    # train test split
    x_train, x_test, y_train, y_test = preprocessing.preprocess_and_split()

    # tạo pipeline chuẩn hóa
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # thực hiện chuẩn hóa
    preprocessor = ColumnTransformer([
        ("num_feature", num_transformer, config_csv.numeric_features),
        ("cat_feature", cat_transformer, config_csv.categorical_features),
        ("binary", "passthrough", config_csv.binary_features)
    ])

    # fit + transform train
    x_train = preprocessor.fit_transform(x_train)

    # chỉ transform test
    x_test = preprocessor.transform(x_test)

    # lưu lại tiền sử lý dữ liệu
    os.makedirs(config_csv.dir_csv_model, exist_ok=True)
    joblib.dump(preprocessor,os.path.join(config_csv.dir_csv_model, "nn_preprocessor.pkl"))


    train(args,x_train, x_test, y_train, y_test)