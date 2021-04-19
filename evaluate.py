import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import multiprocessing
from torchvision import transforms
from tqdm import tqdm
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

dataset_valid = AdvertisementDataset(df=df_valid,
                                     local_image_dir="/home/fatonrekathati/Desktop/newspaper_sections/svd/images",
                                     global_image_dir="/home/fatonrekathati/Desktop/ad_classification/images",
                                     transform=transforms,
                                     mix=True,
                                     global_features=False)

testloader = torch.utils.data.DataLoader(
    dataset_valid, batch_size=8, shuffle=False, num_workers=4)


model = BertEffnetGlobalLocal()
model.to("cuda")
model.load_state_dict(torch.load("saved_models/effbert_globallocal_nometa.pt"))


df_valid = predict(df_valid, testloader, model=model, mix=True)
get_metrics(df_valid)

txt_length = df_valid["text"].apply(lambda x: len(x))
get_metrics(df=df_valid[txt_length > 50])

df_valid["text_length"] = df_valid.loc[:, "text"].apply(lambda x: len(x))


df_preds = df_valid[["id", "dark_id", "part", "page", "label", "pred",
                     "probs", "type", "date", "x", "y", "width", "height", "text_length"]]
df_preds = df_preds.rename(
    columns={"probs": "probs_bertgloballocal_nm", "pred": "preds_bertgloballocal_nm"})
df_preds = df_preds.reset_index(drop=True)

df_valid2 = pd.read_feather("df_preds.feather")

df_valid2["probs_bertgloballocal_nm"] = df_preds["probs_bertgloballocal_nm"]
df_valid2["preds_bertgloballocal_nm"] = df_preds["preds_bertgloballocal_nm"]
# df_valid2["text_length"] = df_preds["text_length"]

df_valid2.to_feather("df_preds.feather")


df_valid = pd.read_feather("df_preds.feather")

df_valid2["text"] = df_valid["text"].values
df_valid = df_valid2.copy()


df_valid[(df_valid["label"] != df_valid["preds_bertgloballocal"])][100:150]

df_valid[txt_length > 6][0:10]
df_valid[["dark_id", "page", "label", "preds_bertgloballocal", "probs_bertgloballocal", "preds_bertglobal", "probs_bertglobal", "preds_global", "probs_global",
          "preds_bertlocal", "probs_bertlocal", "preds_bert", "probs_bert", "text"]][(df_valid["label"] != df_valid["preds_bertgloballocal"]) & (df_valid["text_length"] > 2)][50:100]


df_valid[["id", "dark_id", "page", "label", "preds_bertgloballocal",
          "probs_bertgloballocal"]][(df_valid["label"] != df_valid["preds_bertgloballocal"])]


df_valid[["id", "dark_id", "page", "label", "pred", "probs", "type", "date", "text"]][(
    df_valid["label"] != df_valid["pred"]) & (df_valid["probs"] < 0.001)]

df_valid[["id", "dark_id", "page", "label", "pred", "probs", "type", "date", "text"]][(
    df_valid["label"] != df_valid["pred_bertgloballocal"]) & (df_valid["dark_id"] == 10945535)][0:50]  # page 25
df_valid[["dark_id", "page", "label", "preds_bertgloballocal", "probs_bertgloballocal", "preds_global", "probs_global", "preds_local",
          "probs_local", "preds_bert", "probs_bert", "type", "date", "text"]][(df_valid["dark_id"] == 10945535)][617:670]  # page25


df_valid[["id", "dark_id", "page", "label", "preds_global", "probs_global", "preds_bertglobal",
          "probs_bertglobal", "probs_bertgloballocal", "text"]][(df_valid["dark_id"] == 10945535)][619:665]  # page25

df_valid2.image_path
df_valid["image_path"] = "/home/fatonrekathati/Desktop/newspaper_sections/svd/images/" + \
    df_valid["image_path"]


df_bad = df_valid[(df_valid["dark_id"] == 10945535)][619:665].copy()  # page25


def plot_probs(df, filename, width, height):
    plt.figure(figsize=(width, height))
    img = mpimg.imread(df["image_path"].values[0])
    imgplot = plt.imshow(img)
    plt.axis('off')
    plt.annotate('Full model prob: ' + "{:16.4f}".format(df["probs_bertgloballocal"].values[0]), (
        0, 0), (0, -24), xycoords='axes fraction', textcoords='offset points', va='top', color="tab:blue")
    plt.annotate('Bert+Global prob: ' + "{:13.4f}".format(df["probs_bertglobal"].values[0]), (0, 0), (
        0, -36), xycoords='axes fraction', textcoords='offset points', va='top', color="tab:orange")
    plt.annotate('Global (image) prob: ' + "{:9.4f}".format(df["probs_global"].values[0]), (0, 0), (
        0, -48), xycoords='axes fraction', textcoords='offset points', va='top', color="slategray")
    plt.annotate('Bert+Local prob: ' + "{:15.4f}".format(df["probs_bertlocal"].values[0]), (0, 0), (
        0, -60), xycoords='axes fraction', textcoords='offset points', va='top', color="tab:green")
    plt.annotate('Local (image) prob: ' + "{:11.4f}".format(df["probs_local"].values[0]), (0, 0), (
        0, -72), xycoords='axes fraction', textcoords='offset points', va='top', color="tab:purple")
    plt.annotate('Bert prob: ' + "{:26.4f}".format(df["probs_bert"].values[0]), (0, 0), (
        0, -84), xycoords='axes fraction', textcoords='offset points', va='top', color="red")
    imgplot.figure.savefig("plots/" + filename, aspect="auto", dpi=100)
    plt.clf()
    plt.close("all")

    return imgplot


plot_probs(df=df_bad.iloc[45:46, :], filename="obs46.jpg", width=6, height=5)


df_train.groupby("type").size().agg(
    {"count": lambda x: x, "prop": lambda x: x.sum(level=0)}).unstack(level=0)

df_train.groupby("type").size() / len(df_train)
df_train.groupby("label").size() / len(df_train)

df_valid.groupby("type").size() / len(df_valid)
df_valid.groupby("label").size() / len(df_valid)
