from transformers import pipeline, AutoModelForSequenceClassification, DistilBertTokenizer, AdamW
from sklearn import metrics
import joblib
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np

def train_and_evaluate_model(train_loader,val_loader, model):

    all_actual_labels = []
    all_predicted_labels = []

    optimizer = AdamW(model.parameters(), lr=1e-5)

    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        epoch_actual_labels = []
        epoch_predicted_labels = []

        with torch.no_grad():
            for val_batch in val_loader:
                val_outputs = model(**val_batch)
                val_loss = val_outputs.loss
                val_losses.append(val_loss.item())

                probabilities = torch.nn.functional.softmax(val_outputs.logits, dim=1)
                predicted_labels = torch.argmax(probabilities, dim=1).cpu().numpy()
                
                epoch_actual_labels.extend(val_batch['labels'].cpu().numpy())
                epoch_predicted_labels.extend(predicted_labels)

        average_difference = np.mean(np.abs(np.array(epoch_actual_labels) - np.array(epoch_predicted_labels)))

        print(f'Epoch {epoch + 1}, Training Loss: {loss.item()}, Validation Loss: {sum(val_losses)/len(val_losses)}, Mean Difference: {average_difference}')

        all_actual_labels.append(epoch_actual_labels)
        all_predicted_labels.append(epoch_predicted_labels)

    return model
