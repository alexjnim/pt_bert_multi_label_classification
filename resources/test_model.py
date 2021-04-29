import torch
from resources.build_model import BertClassifier
from sklearn.metrics import classification_report


def test_model(test_dataloader, BERT_MODEL_NAME, num_labels, label_columns):
    # print("\nLoading saved model")
    # model = BertClassifier(
    #     num_labels=num_labels, BERT_MODEL_NAME=BERT_MODEL_NAME, freeze_bert=False
    # )
    # model_dir = "model/model.pt"
    # model.load_state_dict(torch.load(model_dir))
    model_dir = "model/model.pt"
    model = BertClassifier(
        num_labels=num_labels, BERT_MODEL_NAME=BERT_MODEL_NAME, freeze_bert=False
    )
    model.load_state_dict(torch.load(model_dir))
    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions, true_labels = [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Predict
    for batch in test_dataloader:
        # Add batch to GPU
        b_input_ids = batch["input_ids"].to(device)
        b_attention_mask = batch["attention_mask"].to(device)
        b_labels = batch["labels"].to(device)

        # Telling the model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = model(b_input_ids, attention_mask=b_attention_mask)
        # Move logits and labels to CPU
        logits = torch.round(torch.sigmoid(logits))
        logits = logits.detach().numpy().tolist()
        label_ids = b_labels.detach().numpy().tolist()

        # Store predictions and true labels
        predictions.extend(logits)
        true_labels.extend(label_ids)

    print(
        classification_report(
            true_labels,
            predictions,
            target_names=label_columns,
            zero_division=0,
        )
    )
