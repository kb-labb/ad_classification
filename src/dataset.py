import torch
from PIL import Image
import os
import requests
from requests.auth import HTTPBasicAuth


class AdvertisementDataset(torch.utils.data.Dataset):
    """Dataset with cropped images and text from OCR performed on newspapers."""

    def __init__(self, df, local_image_dir, global_image_dir, transform=None, mix=False, global_features=False):
        """
        Args:
            df (DataFrame): DataFrame with text, image path and label annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            mix (bool): True if model uses both global and local image features, False otherwise.
            global_features: True if only global image features are used
        """
        self.df = df
        self.local_image_dir = local_image_dir
        self.global_image_dir = global_image_dir
        self.transform = transform
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers',
                                        'tokenizer',
                                        'KB/bert-base-swedish-cased')  # Download vocabulary from S3 and cache.
        self.mix = mix
        self.global_features = global_features

    def __len__(self):
        return len(self.df)

    def load_image(self, index, is_global):

        if is_global:
            image_path = self.df.iloc[index].image_path_full  # Global
            image_path = os.path.join(self.global_image_dir,
                                      image_path)
        else:
            image_path = self.df.iloc[index].image_path  # Local
            image_path = os.path.join(self.local_image_dir,
                                      image_path)

        image = Image.open(image_path)
        return(image)

    def __getitem__(self, index):

        df_row = self.df.iloc[index]
        label = df_row["label"]

        # Text
        ocr_text = df_row["text"]
        token_info = self.tokenizer.encode_plus(ocr_text,
                                                max_length=64,
                                                truncation=True,
                                                padding="max_length",
                                                return_tensors="pt")

        # Numeric features
        numeric_features = torch.Tensor(
            [df_row["x"], df_row["y"], df_row["height"], df_row["width"]])

        # Categorical embedding features
        categorical_features = torch.LongTensor([df_row["weekday"]])

        if self.mix:
            # Model uses both global and local image features
            image_global = self.load_image(index, is_global=True)
            image_global = self.transform(image_global)

            image_local = self.load_image(index, is_global=False)
            image_local = self.transform(image_local)
            return [image_global, image_local], token_info, numeric_features, categorical_features, label
        else:
            # Model uses global features if self.global_features = True,
            #            local features if self.global_features = False
            image = self.load_image(index, is_global=self.global_features)
            image = self.transform(image)

        return image, token_info, numeric_features, categorical_features, label


# https://datalab.kb.se/dark-11560984/bib13434192_20190405_144231_93_0002.jp2/full/512,512/0/default.jpg
def download_page_image(df, api_password, output_width=512, output_height=512, output_folder="images"):
    """
    Downloads image of the entire newspaper page for given observations.
    """

    df["full_image_url"] = [
        f"{base_url}/full/{output_width},{output_height}/0/default.jpg" for base_url in df["page_image_url"]]
    df["image_path_full"] = df["dark_id"].astype(
        str) + "_part" + df["part"].astype(str) + "_page" + df["page"].astype(str) + ".jpg"

    try:
        os.makedirs(output_folder)
    except OSError as e:
        return df

    # os.makedirs(output_folder, exist_ok=True)

    for i, row in df.iterrows():

        if not os.path.exists(row["image_path_full"]):
            response = requests.get(
                url=row["full_image_url"],
                auth=HTTPBasicAuth("demo", api_password)
            )

            image = response.content
            with open(f"{output_folder}/{row['image_path_full']}", "wb") as image_file:
                image_file.write(image)

        else:
            pass

    return(df)
