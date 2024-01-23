
from torch.utils.data import Dataset, DataLoader
import torch
class CryptoTwittDataset(Dataset):
        def __init__(self, tokenized_data, labels):
            self.tokenized_data = tokenized_data
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                'input_ids': self.tokenized_data['input_ids'][idx],
                'attention_mask': self.tokenized_data['attention_mask'][idx],
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }

        def getDataLoader(data_train, data_test, label_train, label_test,tokenizer):
             
            tokenized_data_train = tokenizer(data_train, padding=True, truncation=True, return_tensors='pt')
            tokenized_data_test = tokenizer(data_test, padding=True, truncation=True, return_tensors='pt')
            

            train_dataset = CryptoTwittDataset(tokenized_data_train, label_train)
            val_dataset = CryptoTwittDataset(tokenized_data_test, label_test)

            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

            return train_loader, val_loader