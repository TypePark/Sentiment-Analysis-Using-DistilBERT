import csv
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, random_split
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertForSequenceClassification, get_linear_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Idk how but I forgot to move the model to the GPU (I figured it out after 2 days of completing the code)

# ----- Preprocessing Section -----
def load_dataset(file_path):
    sentences = []
    labels = []

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            sentence = row[0]  # The sentences are in the first column
            label = int(row[1])  # The labels are in the second column
            sentences.append(sentence)
            labels.append(label)

    return sentences, labels


def preprocess_data(sentences, labels, tokenizer):
    input_ids = []
    attention_masks = []

    for sentence in sentences:
        encoded_inputs = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids.append(encoded_inputs['input_ids'])
        attention_masks.append(encoded_inputs['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0).to(device)
    attention_masks = torch.cat(attention_masks, dim=0).to(device)
    labels = torch.tensor(labels).to(device)

    return input_ids, attention_masks, labels


# Loads the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Loads the dataset
dataset_file = 'Saf.csv'  # Change this if you want to use another dataset
sentences, labels = load_dataset(dataset_file)

# Preprocess the data
input_ids, attention_masks, labels = preprocess_data(sentences, labels, tokenizer)

dataset = TensorDataset(input_ids, attention_masks, labels)

train_ratio = 0.8  # 80% of the data will be used for training

# Calculates the number of samples for training and testing
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])  # Splits the dataset into train and test sets


batch_size = 20  # Reducing the batch size is more effective? (while using CPU)

# Loading the data for training and testing
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)
# ----- Preprocessing Section Ends -----


# ----- Model And Training Loop Section -----
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).to(device)

learning_rate = 0.0001
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # AdamW more effective than Adam for this model
criterion = torch.nn.CrossEntropyLoss()  # This version uses the loss criterion

epochs = 5

# Training loop
for epoch in range(epochs):
    model.train() # Sets the model to training mode
    total_train_loss = 0.0

    for batch_num, batch in enumerate(train_dataloader, 1):
        batch_input_ids, batch_attention_masks, batch_labels = batch

        batch_input_ids = batch_input_ids.to(device)
        batch_attention_masks = batch_attention_masks.to(device)
        batch_labels = batch_labels.to(device)

        model.zero_grad()

        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks, labels=batch_labels)
        logits = outputs.logits
        loss = criterion(logits, batch_labels)  # Computes the loss using the criterion

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() # Accumulates the training loss

        print(f"Epoch: {epoch + 1}/{epochs} | Batch: {batch_num}/{len(train_dataloader)} | Loss: {loss.item()}")

    avg_train_loss = total_train_loss / len(train_dataloader) # Calculates the average training loss for the epoch
    print(f"Epoch: {epoch + 1}/{epochs} | Average Training Loss: {avg_train_loss}")

# Testing loop
model.eval()  # Sets the model to evaluation mode
total_test_loss = 0.0

for batch_num, batch in enumerate(test_dataloader, 1):
    batch_input_ids, batch_attention_masks, batch_labels = batch

    batch_input_ids = batch_input_ids.to(device)
    batch_attention_masks = batch_attention_masks.to(device)
    batch_labels = batch_labels.to(device)

    with torch.no_grad():
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks, labels=batch_labels)
        logits = outputs.logits
        loss = criterion(logits, batch_labels)  # Compute the loss using the criterion

    total_test_loss += loss.item()

    print(f"Batch: {batch_num}/{len(test_dataloader)} | Loss: {loss.item()}")

avg_test_loss = total_test_loss / len(test_dataloader)
print(f"Average Test Loss: {avg_test_loss}")