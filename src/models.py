import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet


class BertResnetClassifier(nn.Module):
    """
    Text + Image + Metadata - model.
    Image features can be either local features (crop of segmented OCR boxes)
    or global features (image of entire page for the observation in question).
    """

    def __init__(self, pretrained=True):
        super().__init__()  # Initialize superclass nn.Module
        if pretrained:
            self.bert = torch.hub.load('huggingface/pytorch-transformers',
                                       'model',
                                       'KB/bert-base-swedish-cased',
                                       output_hidden_states=True)
            self.resnet101 = models.resnet101(pretrained=pretrained)
            self.resnet101.fc = nn.Identity()  # Remove fc layer (classification head)
        else:
            # Load saved models from disk
            pass

        # (, 2048) (image) + (, 768) (text) + (, 7) (numerical) = (, 2823)
        self.fc = nn.Sequential(
            nn.Linear(2048 + 768 + 7, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1)
        )

        self.categorical_embedder = nn.Embedding(
            8, 3, padding_idx=0)  # weekday embedding

    def forward(self, image, token_ids, type_ids, mask, numeric_features, categorical_features):
        # image.unsqueeze_(0) # Add batch dimension to image tensor
        image_embedding = self.resnet101(image)
        hidden_states = self.bert(
            token_ids, token_type_ids=type_ids, attention_mask=mask)
        categorical_embedding = self.categorical_embedder(
            categorical_features).squeeze_(dim=1)

        combined_embedding = torch.cat([image_embedding,
                                        hidden_states[0][:, 0, :],
                                        numeric_features,
                                        categorical_embedding],
                                       dim=1)
        output = self.fc(combined_embedding)

        return output


class BertEfficientnetClassifier(nn.Module):
    """
    Text + Image + Metadata - model.
    Image features can be either local features (crop of segmented OCR boxes)
    or global features (image of entire page for the observation in question).
    EfficientNet backbone for the image model.
    """

    def __init__(self, pretrained=True):
        super().__init__()  # Initialize superclass nn.Module
        if pretrained:
            self.bert = torch.hub.load('huggingface/pytorch-transformers',
                                       'model',
                                       'KB/bert-base-swedish-cased',
                                       output_hidden_states=True)
            self.effnet = EfficientNet.from_pretrained(
                'efficientnet-b2', include_top=False)
        else:
            # Load saved models from disk
            pass

        self.fc = nn.Sequential(
            nn.Linear(1408 + 768 + 7, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1)
        )

        self.categorical_embedder = nn.Embedding(
            8, 3, padding_idx=0)  # weekday embedding

    def forward(self, image, token_ids, type_ids, mask, numeric_features, categorical_features):
        # image.unsqueeze_(0) # Add batch dimension to image tensor
        image_embedding = self.effnet(image)  # (1, 1408, 1, 1)
        image_embedding.squeeze_(-1)
        image_embedding.squeeze_(-1)
        hidden_states = self.bert(
            token_ids, token_type_ids=type_ids, attention_mask=mask)
        categorical_embedding = self.categorical_embedder(
            categorical_features).squeeze_(dim=1)

        # (1, 1408) (image) + (1, 768) (text) + (1, 4) + (1, 3)
        combined_embedding = torch.cat(
            [image_embedding, hidden_states[0][:, 0, :], numeric_features, categorical_embedding], dim=1)

        output = self.fc(combined_embedding)

        return output


class BertClassifier(nn.Module):
    """
    Text + Metadata - model.
    """

    def __init__(self, pretrained=True):
        super().__init__()  # Initialize superclass nn.Module
        if pretrained:
            self.bert = torch.hub.load('huggingface/pytorch-transformers',
                                       'model',
                                       'KB/bert-base-swedish-cased',
                                       output_hidden_states=True)
        else:
            # Load saved models from disk
            pass

        self.fc = nn.Sequential(
            nn.Linear(768 + 3, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1)
        )

        self.categorical_embedder = nn.Embedding(
            8, 3, padding_idx=0)  # weekday embedding

    def forward(self, image, token_ids, type_ids, mask, numeric_features, categorical_features):

        hidden_states = self.bert(
            token_ids, token_type_ids=type_ids, attention_mask=mask)
        categorical_embedding = self.categorical_embedder(
            categorical_features).squeeze_(dim=1)

        # (1, 1408) (image) + (1, 768) (text) + (1, 4) + (1, 3)
        combined_embedding = torch.cat(
            [hidden_states[0][:, 0, :],
             # numeric_features,
             categorical_embedding], dim=1)

        output = self.fc(combined_embedding)

        return output


class EffnetClassifier(nn.Module):
    """
    Image + metadata model.
    Image features can be either local features or global features.
    """

    def __init__(self, pretrained=True):
        super().__init__()  # Initialize superclass nn.Module
        if pretrained:
            self.effnet = EfficientNet.from_pretrained(
                'efficientnet-b2', include_top=False)
        else:
            # Load saved models from disk
            pass

        self.fc = nn.Sequential(
            nn.Linear(1408 + 3, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1)
        )

        self.categorical_embedder = nn.Embedding(
            8, 3, padding_idx=0)  # weekday embedding

    def forward(self, image, token_ids, type_ids, mask, numeric_features, categorical_features):
        # image.unsqueeze_(0) # Add batch dimension to image tensor
        image_embedding = self.effnet(image)  # (1, 1408, 1, 1)
        image_embedding.squeeze_(-1)
        image_embedding.squeeze_(-1)
        categorical_embedding = self.categorical_embedder(
            categorical_features).squeeze_(dim=1)

        combined_embedding = torch.cat(
            [image_embedding,
             # numeric_features,
             categorical_embedding],
            dim=1)  # (1, 1408) (image) + (1, 768) (text) + (1, 4) + (1, 3)

        output = self.fc(combined_embedding)

        return output


class BertEffnetGlobalLocal(nn.Module):
    """
    Text + Image (local) + Image (global) + Metadata - model.
    Trains 2 CNNs in parallel. One CNN takes cropped image of segmented OCR region
    as input (local), the other takes the image of the entire newspaper page (global).
    """

    def __init__(self, pretrained=True):
        super().__init__()  # Initialize superclass nn.Module
        if pretrained:
            self.bert = torch.hub.load('huggingface/pytorch-transformers',
                                       'model',
                                       'KB/bert-base-swedish-cased',
                                       output_hidden_states=True)
            self.effnet_global = EfficientNet.from_pretrained(
                'efficientnet-b2', include_top=False)
            self.effnet_local = EfficientNet.from_pretrained(
                'efficientnet-b2', include_top=False)
        else:
            # Load saved models from disk
            pass

        self.fc = nn.Sequential(
            nn.Linear(1408 + 1408 + 768 + 3, 512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 1)
        )

        self.categorical_embedder = nn.Embedding(
            8, 3, padding_idx=0)  # weekday embedding

    def forward(self, image, token_ids, type_ids, mask, numeric_features, categorical_features):
        # image.unsqueeze_(0) # Add batch dimension to image tensor
        image_embedding_global = self.effnet_global(
            image[0])  # (1, 1408, 1, 1)
        image_embedding_local = self.effnet_local(image[1])  # (1, 1408, 1, 1)
        image_embedding_global.squeeze_(-1)
        image_embedding_global.squeeze_(-1)
        image_embedding_local.squeeze_(-1)
        image_embedding_local.squeeze_(-1)
        hidden_states = self.bert(
            token_ids, token_type_ids=type_ids, attention_mask=mask)
        categorical_embedding = self.categorical_embedder(
            categorical_features).squeeze_(dim=1)

        combined_embedding = torch.cat([image_embedding_global,
                                       image_embedding_local,
                                       hidden_states[0][:, 0, :],
                                       # numeric_features,
                                        categorical_embedding],
                                       dim=1)  # (1, 1408) (image) + (1, 768) (text) + (1, 4) + (1, 3)
        output = self.fc(combined_embedding)

        return output
