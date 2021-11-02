import time
import random
import numpy as np
import torch
from utils.helper_functions import format_time
from transformers import AdamW, get_linear_schedule_with_warmup


def train_model(
    model, train_dataloader, val_dataloader=None, epochs=5, evaluation=False
):
    """Train and validate the BertClassifier model."""
    training_stats = []
    train_loss_set = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer, scheduler = build_optimizer_scheduler(
        model=model, epochs=epochs, train_dataloader=train_dataloader
    )
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        print(
            f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}"
        )
        print("-" * 70)
        t0 = time.time()
        t0_epoch, t0_batch = time.time(), time.time()
        total_loss, batch_loss, batch_counts = 0, 0, 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            b_input_ids = batch["input_ids"].to(device)
            b_attention_mask = batch["attention_mask"].to(device)
            b_labels = batch["labels"].to(device)
            # b_attention_mask, b_labels = tuple(t.to(device) for t in batch)

            model.zero_grad()
            # i think this should be
            optimizer.zero_grad()

            loss = model(
                input_ids=b_input_ids,
                attention_mask=b_attention_mask,
                labels=b_labels,
            )
            batch_loss += loss.item()
            total_loss += loss.item()
            train_loss_set.append(loss.item())

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch
                print(
                    f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}"
                )
                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)

        print("-" * 70)

        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            avg_val_loss, avg_val_accuracy, validation_time = evaluate(
                model, val_dataloader
            )
            time_elapsed = time.time() - t0_epoch
            print(
                f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {avg_val_loss:^10.6f} | {avg_val_accuracy:^9.2f} | {time_elapsed:^9.2f}"
            )
            print("-" * 70)

        # save model
        if (
            len(training_stats) == 0
            or training_stats[-1]["Valid. Loss"] > avg_train_loss
        ):
            model_dir = "model/model.pt"
            torch.save(model.state_dict(), model_dir)

        training_stats.append(
            {
                "epoch": epoch_i + 1,
                "Training Loss": avg_train_loss,
                "Valid. Loss": avg_val_loss,
                "Valid. Accur.": avg_val_accuracy,
                "Training Time": training_time,
                "Validation Time": validation_time,
            }
        )

    print("\n")
    print("Training complete!")
    return model, training_stats, train_loss_set


def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    t0 = time.time()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Tracking variables
    avg_val_accuracy = []
    avg_val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        b_input_ids = batch["input_ids"].to(device)
        b_attention_mask = batch["attention_mask"].to(device)
        b_labels = batch["labels"].to(device)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attention_mask)

        # Compute loss
        loss_func = model.loss_func
        predictions = torch.sigmoid(logits)
        loss = loss_func(
            predictions.view(-1, model.num_labels), b_labels.view(-1, model.num_labels)
        )
        avg_val_loss.append(loss.item())

        # Get the predictions
        preds = torch.round(predictions)

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        avg_val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    avg_val_loss = np.mean(avg_val_loss)
    avg_val_accuracy = np.mean(avg_val_accuracy)
    validation_time = format_time(time.time() - t0)
    return avg_val_loss, avg_val_accuracy, validation_time


def build_optimizer_scheduler(model, epochs, train_dataloader):

    # setting custom optimization parameters for huggingface model and implement a scheduler here as well.
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=5e-5,  # Default learning rate
        eps=1e-8,  # Default epsilon value
    )

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,  # Default value
        num_training_steps=total_steps,
    )

    return optimizer, scheduler
