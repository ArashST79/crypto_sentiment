from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score
from utilities import Vader_senti
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
class Evaluator:
    def __init__(self, model, tokenizer, data):
        self.model = model
        self.tokenizer = tokenizer
        self.data = data
        self.predicted_labels = []
        self.actual_labels = []
        self.prediction_outputs = []
    def evaluate_model_crypto_bert(self):
        self.actual_labels = self.data['entities.sentiment.basic']

        for idx, row in tqdm(self.data.iterrows(), total=len(self.data), desc="CryptoBERT Sentiment Analysis"):
            text = row['body']
            # Tokenize and get predictions
            inputs = self.tokenizer(text, return_tensors='pt')
            outputs = self.model(**inputs)
            prediction_output = torch.nn.functional.softmax(outputs.logits, dim=-1)
            if(prediction_output[0][2] > prediction_output[0][0]):
                self.predicted_labels.append("Bullish")
            else:
                self.predicted_labels.append("Bearish")
            self.prediction_outputs.append(prediction_output)
        accuracy = accuracy_score(self.actual_labels, self.predicted_labels)
        return accuracy
    
    def evaluate_model_crypto_bert2(self):
        self.actual_labels = self.data['entities.sentiment.basic']

        for idx, row in tqdm(self.data.iterrows(), total=len(self.data), desc="CryptoBERT 2 Sentiment Analysis"):
            text = row['body']
            # Tokenize and get predictions
            inputs = self.tokenizer(text, return_tensors='pt')
            outputs = self.model(**inputs)
            prediction_output = torch.nn.functional.softmax(outputs.logits, dim=-1)
            if(prediction_output[0][1] > prediction_output[0][0]):
                self.predicted_labels.append("Bullish")
            else:
                self.predicted_labels.append("Bearish")
            self.prediction_outputs.append(prediction_output)
        accuracy = accuracy_score(self.actual_labels, self.predicted_labels)
        return accuracy
    
    def evaluate_model_fin_bert(self):
    
        self.actual_labels = self.data['entities.sentiment.basic']

        for idx, row in tqdm(self.data.iterrows(), total=len(self.data), desc="FinBERT Sentiment Analysis"):
            text = row['body']
            # Tokenize and get predictions
            inputs = self.tokenizer(text, return_tensors='pt')
            outputs = self.model(**inputs)
            prediction_output = torch.nn.functional.softmax(outputs.logits, dim=-1)
            if(prediction_output[0][0] > prediction_output[0][1]):
                self.predicted_labels.append("Bullish")
            else:
                self.predicted_labels.append("Bearish")
            self.prediction_outputs.append(prediction_output)
        accuracy = accuracy_score(self.actual_labels, self.predicted_labels)
        return accuracy
    
    def evaluate_vader(self):
        self.actual_labels = self.data['entities.sentiment.basic']

        # Use tqdm to show a progress bar during iteration
        for index, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Vader Sentiment Analysis"):
            text = row['body']
            # Assuming vader_senti returns a tuple (negativity, neutrality, positivity, sum)
            sentiment_scores = Vader_senti(text)
            
            # Determine the predicted label based on sentiment scores
            self.predicted_label = 'Bearish' if sentiment_scores[0] > sentiment_scores[2] else 'Bullish'
            self.predicted_labels.append(self.predicted_label)

        accuracy = accuracy_score(self.actual_labels, self.predicted_labels)
        return accuracy

    def show_bad_samples(self,num):
        import random

        # Set a seed for reproducibility
        random.seed(42)

        # Find indices of instances where the model predicted Bullish for Bearish comments
        mismatch_indices = [idx for idx, (actual, predicted) in enumerate(zip(self.actual_labels, self.predicted_labels)) if actual == 'Bearish' and predicted == 'Bullish']

        # Randomly select 20 indices from the mismatched instances
        random_sample_indices = random.sample(mismatch_indices, min(num, len(mismatch_indices)))

        for idx in random_sample_indices:
            actual = self.actual_labels.iloc[idx]  # Use .iloc to access values by index
            predicted = self.predicted_labels[idx]
            text = self.data.iloc[idx]['body']

            print(f"Actual Label: {actual}")
            print(f"Predicted Label: {predicted}")
            print(f"Text: {text}")
            print("Prediction Confidency \n" + str(self.prediction_outputs[idx]))
            print("\n---------------------\n")
            
    def show_confusion_matrix(self):
        

        # Assuming cm is your confusion matrix
        class_names = ['Bearish', 'Bullish']
        cm = confusion_matrix(self.actual_labels, self.predicted_labels, labels=class_names)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

        # Customize the color map
        cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
        disp = disp.plot(cmap=cmap)

        plt.title("Confusion Matrix")
        plt.show()
        