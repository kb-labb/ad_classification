import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import multiprocessing
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_curve
from plotnine import ggplot, aes, geom_line, geom_abline, ggtitle
from itertools import product

from src.dataset import AdvertisementDataset, download_page_image
from src.models import BertResnetClassifier, BertEfficientnetClassifier, BertClassifier, EffnetClassifier, BertEffnetGlobalLocal
from src.evaluation import predict, get_metrics

pd.set_option('display.max_colwidth', 80)
pd.options.display.float_format = '{:.4f}'.format

df = pd.read_parquet("data/svd_ads.parquet")
df = df[(df["label"] != "mixed") & (~df["label"].isnull())]
df = df.rename(columns={"content": "text"})
df.loc[df["text"].isnull(), "text"] = ""

df.loc[df["label"] == "editorial", "label"] = 0
df.loc[df["label"] == "ad", "label"] = 1
df["label"] = df["label"].astype("int8")

# Download full newspaper page
pw = open("../api_credentials.txt", 'r').readlines()
df_split = np.array_split(df, 16, axis=0)
pool = multiprocessing.Pool()
df = pool.starmap(download_page_image, product(df_split, pw)
                  )  # starmap for adding optional args
df = pd.concat(df)
pool.close()

# Time split
df_train = df[1:68034]  # When using full page model
df_valid = df[68034:]

transforms = transforms.Compose([
    transforms.Resize(size=(260, 260),
                      interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset_train = AdvertisementDataset(df=df_train,
                                     local_image_dir="/home/fatonrekathati/Desktop/newspaper_sections/svd/images",
                                     global_image_dir="/home/fatonrekathati/Desktop/ad_classification/images",
                                     transform=transforms,
                                     mix=False,
                                     global_features=False)

dataset_valid = AdvertisementDataset(df=df_valid,
                                     local_image_dir="/home/fatonrekathati/Desktop/newspaper_sections/svd/images",
                                     global_image_dir="/home/fatonrekathati/Desktop/ad_classification/images",
                                     transform=transforms,
                                     mix=False,
                                     global_features=False)

dataloader = torch.utils.data.DataLoader(
    dataset_train, batch_size=16, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(
    dataset_valid, batch_size=8, shuffle=False, num_workers=4)


model = BertClassifier()
model.to("cuda")
# model.load_state_dict(torch.load("saved_models/effbert_full_page.pt"))

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=0.00002)  # 0.00002


def lambda1(epoch):
    return 0.65 ** epoch


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

# Train
log_list = []
for epoch in range(1):
    print(f"epoch: {epoch + 1} started")
    running_loss = 0
    for i, batch in enumerate(tqdm(dataloader)):

        images = batch[0]
        # images[0] = images[0].to("cuda")
        # images[1] = images[1].to("cuda")
        texts = batch[1]
        numeric_features = batch[2]
        categorical_features = batch[3]
        labels = batch[4].to("cuda")
        optimizer.zero_grad()
        output = model(image=images,
                       token_ids=texts["input_ids"].squeeze(dim=1).to("cuda"),
                       type_ids=texts["token_type_ids"].squeeze(
                           dim=1).to("cuda"),
                       mask=texts["attention_mask"].squeeze(dim=1).to("cuda"),
                       numeric_features=numeric_features.to("cuda"),
                       categorical_features=categorical_features.to("cuda"))

        labels = labels.unsqueeze(1).type_as(
            output)  # (8) -> (8, 1) and long to float
        loss = loss_fn(output, labels)

        running_loss += loss.item()

        if i % 50 == 49:
            print(f"iter: {i+1}, loss: {running_loss/50:.3f}")
            log_list.append({"iter": i+1, "loss": running_loss/50})
            running_loss = 0

        loss.backward()
        optimizer.step()

    scheduler.step()

    # Predict
    df_valid = predict(df_valid, testloader, model=model, mix=False)
    get_metrics(df_valid)


def plot_roc_auc(df):
    fpr, tpr, threshold = roc_curve(df["label"], df["probs"])
    df_plot = pd.DataFrame(dict(fpr=fpr, tpr=tpr))

    p = (
        ggplot(data=df_plot)
        + aes(x="fpr", y="tpr")
        + geom_line()
        + geom_abline(linetype="dashed")
        + ggtitle("ROC AUC for newspaper ad classification")
    )

    return p


df_valid2 = df_valid.copy()
df_valid2["pred"] = 1
df_valid2.loc[df_valid["probs"] < 0.95, "pred"] = 0

get_metrics(df_valid)
get_metrics(df_valid[df_valid["type"] == "Image"])
get_metrics(df_valid2)

p = plot_roc_auc(df_valid2)
p.save(filename="roc_auc.png")

# torch.save(model.state_dict(), "saved_models/bertclassifier_nometa.pt")
