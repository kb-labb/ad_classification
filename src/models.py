import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet

class BertResnetClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__() # Initialize superclass nn.Module
        if pretrained:
            self.bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'KB/bert-base-swedish-cased', output_hidden_states=True)
            self.resnet101 = models.resnet101(pretrained=pretrained)
            self.resnet101.fc = nn.Identity() # Remove fc layer (classification head)
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

        self.categorical_embedder = nn.Embedding(8, 3, padding_idx=0) # weekday embedding

    def forward(self, image, token_ids, type_ids, mask, numeric_features, categorical_features):
        # image.unsqueeze_(0) # Add batch dimension to image tensor
        image_embedding = self.resnet101(image)
        hidden_states = self.bert(token_ids, token_type_ids=type_ids, attention_mask=mask)
        categorical_embedding = self.categorical_embedder(categorical_features).squeeze_(dim=1)
        
        output_embedding = torch.cat([image_embedding, hidden_states[0][:,0,:]], dim=1)
        combined_embedding = torch.cat([image_embedding, 
                                        hidden_states[0][:,0,:], 
                                        numeric_features, 
                                        categorical_embedding], 
                                        dim=1)
        output = self.fc(combined_embedding)

        return output


class BertEfficientnetClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__() # Initialize superclass nn.Module
        if pretrained:
            self.bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'KB/bert-base-swedish-cased', output_hidden_states=True)
            self.effnet = EfficientNet.from_pretrained('efficientnet-b2', include_top=False)
        else:
            # Load saved models from disk
            pass

        self.fc = nn.Sequential(
            nn.Linear(1408 + 768 + 7, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1)
        )

        self.categorical_embedder = nn.Embedding(8, 3, padding_idx=0) # weekday embedding

    def forward(self, image, token_ids, type_ids, mask, numeric_features, categorical_features):
        # image.unsqueeze_(0) # Add batch dimension to image tensor
        image_embedding = self.effnet(image) # (1, 1408, 1, 1)
        image_embedding.squeeze_(-1)
        image_embedding.squeeze_(-1)
        hidden_states = self.bert(token_ids, token_type_ids=type_ids, attention_mask=mask)
        categorical_embedding = self.categorical_embedder(categorical_features).squeeze_(dim=1)
        
        combined_embedding = torch.cat([image_embedding, 
                                        hidden_states[0][:,0,:], 
                                        numeric_features, 
                                        categorical_embedding], 
                                        dim=1) # (1, 1408) (image) + (1, 768) (text) + (1, 4) + (1, 3)
        output = self.fc(combined_embedding)

        return output