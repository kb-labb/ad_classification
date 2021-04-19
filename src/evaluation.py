import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve


def predict(df, testloader, model, mix=False):
    """
    Args:
        mix (bool): True if both global and local image features were used.
        False otherwise.
    """
    probs_list = []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(testloader)):
            if mix:
                images = batch[0]
                images[0] = images[0].to("cuda")
                images[1] = images[1].to("cuda")
            else:
                images = batch[0].to("cuda")
            texts = batch[1]
            numeric_features = batch[2]
            categorical_features = batch[3]
            output = model(image=images,
                           token_ids=texts["input_ids"].squeeze(
                               dim=1).to("cuda"),
                           type_ids=texts["token_type_ids"].squeeze(
                               dim=1).to("cuda"),
                           mask=texts["attention_mask"].squeeze(
                               dim=1).to("cuda"),
                           numeric_features=numeric_features.to("cuda"),
                           categorical_features=categorical_features.to("cuda"))

            probs_list += torch.sigmoid(output).tolist()

        df["probs"] = sum(probs_list, [])
        df.sort_values(by="probs", ascending=False)

        df["pred"] = 1
        df.loc[df["probs"] < 0.5, "pred"] = 0

    return df


def get_metrics(df):
    metrics = {
        "accuracy": accuracy_score(y_true=df["label"], y_pred=df["pred"]),
        "f1": f1_score(y_true=df["label"], y_pred=df["pred"]),
        "precision": precision_score(y_true=df["label"], y_pred=df["pred"]),
        # Sensitivity (# correct ad preds / # true ads)
        "sensitivity": recall_score(y_true=df["label"], y_pred=df["pred"]),
        # Specificity (# correct editorial preds / # true editorial)
        "specificity": recall_score(y_true=df["label"], y_pred=df["pred"], pos_label=0),
        "roc_auc": roc_auc_score(y_true=df["label"], y_score=df["probs"])
    }

    print(metrics)
    return metrics
