import argparse
from src_image import dataset
from configs import config_image
from src_image.model import DiabeticRetinopathy, DiabeticRetinopathyPretrain
import torch
import os
import torch.nn as nn
import shutil
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW,SGD
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def get_args():
    parser = argparse.ArgumentParser(description="train model")
    parser.add_argument("--batch_size","-b",type=int,default=config_image.batch_size,help="number batch size")
    parser.add_argument("--image_size","-i",type=int,default=config_image.image_size,help="number image size")
    parser.add_argument("--epochs","-e",type=int,default=config_image.epochs,help="number epoch")
    parser.add_argument("--learning_rate","-r",type=float,default=config_image.learning_rate,help="number learning rate")
    parser.add_argument("--weight_decay","-w",type=float,default=config_image.weight_decay,help="number weight decay")
    parser.add_argument("--momentom","-m",type=float,default=config_image.momentum,help="number momentom")
    parser.add_argument("--trained_models","-t",type=str,default=config_image.path_model,help="model path ")
    parser.add_argument("--logging","-l",type=str,default=config_image.path_tensorboard,help="tensorboard path")
    parser.add_argument("--checkpoint","-c",type=str,default=None,help="checkpoint")

    args = parser.parse_args()
    return args


def train(args):
    # chuyển máy sang cpu
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    train_dataloader, test_dataloader = dataset.dataloader(batch_size=args.batch_size,image_size=args.image_size)

    # check thư mục tensorboard cũ xóa bỏ nó đi
    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)

    # kiểm tra xem có thư mục lưu model chưa nếu chưa có thì tạo
    if not os.path.isdir(args.trained_models):
        os.makedirs(args.trained_models)

    # tạo tensorboard
    writer = SummaryWriter(args.logging)

    # khởi tạo model
    model = DiabeticRetinopathy().to(device)

    # khởi tạo hàm loss
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    # ko có cái
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    num_iteration = len(train_dataloader)

    #load model cũ
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 0
        best_acc = 0

    for epoch in range(start_epoch,args.epochs):
        model.train()
        progress_bar = tqdm(train_dataloader,colour="cyan")
        for iter,(images,labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            # cho vào model
            outputs = model(images)

            # tính loss
            loss_value = criterion(outputs,labels)

            writer.add_scalar("train/loss",loss_value.item(),epoch*num_iteration+iter)
            progress_bar.set_description("Epoch{}/{}  , iteration {}/{} , loss {:.3f}".format(epoch+1,args.epochs,iter+1,num_iteration,loss_value.item()))

            #backward
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        model.eval()
        progress_bar = tqdm(test_dataloader, colour="red")
        all_labels = []
        all_predictions = []
        for iter, (images, labels) in enumerate(progress_bar):
            all_labels.extend(labels)

            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                predictions = model(images)
                indices = torch.argmax(predictions.cpu(),dim=1)
                all_predictions.extend(indices)

        scheduler.step() # cũ ko có cái này

        all_labels = [label.item() for label in all_labels]
        all_predictions = [prediction.item() for prediction in all_predictions ]

        accuracy = accuracy_score(all_labels, all_predictions)
        print("Epoch : {}, acuaracy : {} ".format(epoch+1,accuracy))
        writer.add_scalar("val/Accuracy", accuracy, epoch)

        # save model
        checkpoint = {
            "epoch": epoch+1,
            "best_acc":accuracy,
            "model":model.state_dict(),
            "optimizer":optimizer.state_dict()
        }
        torch.save(checkpoint, "{}/last_cnn.pt".format(args.trained_models))
        # save best model ,learning rate ,epochs
        if accuracy > best_acc:
            checkpoint = {
                "epoch": epoch + 1,
                "best_acc": best_acc,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(checkpoint, "{}/best_cnn.pt".format(args.trained_models))
            best_acc = accuracy
    writer.close()

if __name__ == '__main__':
    args = get_args()
    train(args)