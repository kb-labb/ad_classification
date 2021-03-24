import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import multiprocessing
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from plotnine import ggplot, aes, geom_line, geom_abline, ggtitle
from itertools import product

from src.dataset import AdvertisementDataset, download_page_image
from src.models import BertResnetClassifier, BertEfficientnetClassifier

pd.set_option('display.max_colwidth', 70)
pd.options.display.float_format = '{:.4f}'.format

df = pd.read_parquet("data/svd_ads.parquet")
df = df[(df["label"] != "mixed") & (~df["label"].isnull())]
df = df.rename(columns={"content": "text"})
df.loc[df["text"].isnull(), "text"] = ""

df.loc[df["label"] == "editorial", "label"] = 0
df.loc[df["label"] == "ad", "label"] = 1
df["label"] = df["label"].astype("int8")

# Download full newspaper page
pw = open("../api_credentials.txt",'r').readlines()
df_split = np.array_split(df, 16, axis=0)
pool = multiprocessing.Pool()
df = pool.starmap(download_page_image, product(df_split, pw)) # starmap for adding optional args
df = pd.concat(df)
pool.close()

# df["image_path"] = "images/" + df["image_path_full"]

np.random.seed(1337)
df["training_set"] = np.random.choice([0, 1], size=len(df), p=[0.2, 0.8])
df_train = df[df["training_set"] == 1]
df_valid = df[df["training_set"] == 0]


def predict(df, testloader, model):
    probs_list = []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(testloader)):
            images = batch[0]
            texts = batch[1]
            numeric_features = batch[2]
            categorical_features = batch[3]
            output = model(image=images.to("cuda"),
                        token_ids=texts["input_ids"].squeeze(dim=1).to("cuda"),
                        type_ids=texts["token_type_ids"].squeeze(dim=1).to("cuda"),
                        mask=texts["attention_mask"].squeeze(dim=1).to("cuda"),
                        numeric_features=numeric_features.to("cuda"),
                        categorical_features=categorical_features.to("cuda"))

            probs_list += torch.sigmoid(output).tolist()

        df["probs"] = sum(probs_list, [])
        df.sort_values(by = "probs", ascending=False)

        df["pred"] = 1
        df.loc[df["probs"] < 0.5, "pred"] = 0

    return df

def get_metrics(df):
    metrics = {
        "accuracy": accuracy_score(y_true=df["label"], y_pred=df["pred"]),
        "f1": f1_score(y_true=df["label"], y_pred=df["pred"]),
        "precision": precision_score(y_true=df["label"], y_pred=df["pred"]),
        "sensitivity": recall_score(y_true=df["label"], y_pred=df["pred"]), # Sensitivity (# correct ad preds / # true ads)
        "specificity": recall_score(y_true=df["label"], y_pred=df["pred"], pos_label=0), # Specificity (# correct editorial preds / # true editorial)
        "roc_auc": roc_auc_score(y_true=df["label"], y_score=df["probs"])
    }

    print(metrics)
    return metrics


transforms = transforms.Compose([
    transforms.Resize(size=(260, 260), 
                      interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# "/home/fatonrekathati/Desktop/ad_classification"
# "/home/fatonrekathati/Desktop/newspaper_sections/svd/images"
dataset_train = AdvertisementDataset(df=df_train, 
                                     root_dir="/home/fatonrekathati/Desktop/newspaper_sections/svd/images",
                                     transform=transforms)

dataset_valid = AdvertisementDataset(df=df_valid, 
                                     root_dir="/home/fatonrekathati/Desktop/newspaper_sections/svd/images",
                                     transform=transforms)

dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=24, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(dataset_valid, batch_size=8, shuffle=False, num_workers=4)


model = BertEfficientnetClassifier()
model.to("cuda")

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00004) # 0.00002
lambda1 = lambda epoch: 0.65 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

# Train
log_list = []
for epoch in range(3):
    print(f"epoch: {epoch + 1} started")
    running_loss = 0
    for i, batch in enumerate(tqdm(dataloader)):

        images = batch[0]
        texts = batch[1]
        numeric_features = batch[2]
        categorical_features = batch[3]
        labels = batch[4].to("cuda")
        optimizer.zero_grad()
        output = model(image=images.to("cuda"),
                    token_ids=texts["input_ids"].squeeze(dim=1).to("cuda"),
                    type_ids=texts["token_type_ids"].squeeze(dim=1).to("cuda"),
                    mask=texts["attention_mask"].squeeze(dim=1).to("cuda"),
                    numeric_features=numeric_features.to("cuda"),
                    categorical_features=categorical_features.to("cuda"))

        labels = labels.unsqueeze(1).type_as(output) # (8) -> (8, 1) and long to float
        loss = loss_fn(output, labels)

        running_loss += loss.item()

        if i % 50 == 49:    # print every 2000 mini-batches
            print(f"iter: {i+1}, loss: {running_loss/50:.3f}")
            log_list.append({"iter": i+1, "loss": running_loss/50})
            running_loss = 0

        loss.backward()
        optimizer.step()

    scheduler.step()

    # Predict
    df_valid = predict(df_valid, testloader, model=model)
    get_metrics(df_valid)


def plot_roc_auc(df):
    fpr, tpr, threshold = roc_curve(df["label"], df["probs"])
    df_plot = pd.DataFrame(dict(fpr = fpr, tpr = tpr))

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
get_metrics(df_valid2)

p = plot_roc_auc(df_valid2)
p.save(filename="roc_auc.png")

txt_length = df_valid["text"].apply(lambda x: len(x))

get_metrics(df=df_valid2[txt_length > 20])
get_metrics(df=df_valid[txt_length > 30])

df_valid2[txt_length > 6][0:10]
df_valid2[(df_valid2["label"] != df_valid2["pred"]) & (txt_length > 2)][150:200]

df_valid2.loc[:,"probs"][txt_length > 2][0:50]

sum((df_valid["pred"] == 1) & (df_valid["label"] == 1))/sum(df_valid["label"] == 1)