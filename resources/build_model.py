import torch
import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(self, num_labels: int, BERT_MODEL_NAME, freeze_bert=False):
        super().__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)

        # Specify hidden size of BERT, hidden size of our classifier, and number of labels to classify
        D_in, H, D_out = self.bert.config.hidden_size, 50, num_labels

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(H, D_out),
        )

        self.loss_func = nn.BCEWithLogitsLoss()

        if freeze_bert:
            print("freezing bert parameters")
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        logits = self.classifier(last_hidden_state_cls)

        if labels is not None:
            predictions = torch.sigmoid(logits)
            loss = self.loss_func(
                predictions.view(-1, self.num_labels), labels.view(-1, self.num_labels)
            )
            return loss
        else:
            return logits
