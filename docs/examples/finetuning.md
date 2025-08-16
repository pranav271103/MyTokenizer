# Fine-tuning Examples

This guide demonstrates how to fine-tune pre-trained language models using the tokenizer.

## Table of Contents
- [Fine-tuning BERT](#fine-tuning-bert)
- [Fine-tuning RoBERTa](#fine-tuning-roberta)
- [Fine-tuning GPT-2](#fine-tuning-gpt-2)
- [Custom Dataset with DataLoader](#custom-dataset-with-dataloader)
- [Training Loop with PyTorch Lightning](#training-loop-with-pytorch-lightning)
- [Saving and Loading Fine-tuned Models](#saving-and-loading-fine-tuned-models)

## Fine-tuning BERT

```python
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW
from tokenizer import Tokenizer

# Load pre-trained tokenizer
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

# Prepare dataset
texts = ["This is a positive example.", "This is a negative example."]
labels = [1, 0]

# Tokenize inputs
inputs = tokenizer.encode_batch(texts, padding=True, truncation=True, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
labels = torch.tensor(labels)

# Create DataLoader
dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
dataloader = DataLoader(dataset, batch_size=2)

# Load pre-trained model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
model.train()
for epoch in range(3):  # Number of epochs
    for batch in dataloader:
        batch_input_ids, batch_attention_mask, batch_labels = batch
        
        # Forward pass
        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            labels=batch_labels
        )
        
        # Backward pass
        loss = outputs.loss
        loss.backward()
        
        # Update weights
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"Loss: {loss.item()}")

# Save the fine-tuned model
model.save_pretrained("models/finetuned_bert")
tokenizer.save("models/finetuned_bert/tokenizer.model")
```

## Fine-tuning RoBERTa

```python
from transformers import RobertaForSequenceClassification, AdamW
from tokenizer import Tokenizer
import torch

# Load RoBERTa tokenizer
tokenizer = Tokenizer.from_pretrained("roberta-base")

# Prepare dataset
texts = ["This is the first example.", "This is the second example."]
labels = [1, 0]

# Tokenize inputs
inputs = tokenizer.encode_batch(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

# Create DataLoader
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
labels = torch.tensor(labels)

dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
dataloader = DataLoader(dataset, batch_size=2)

# Load RoBERTa model
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# Set up optimizer and training parameters
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
model.train()
for epoch in range(3):
    for batch in dataloader:
        batch_input_ids, batch_attention_mask, batch_labels = batch
        
        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            labels=batch_labels
        )
        
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save the model
model.save_pretrained("models/finetuned_roberta")
tokenizer.save("models/finetuned_roberta/tokenizer.model")
```

## Fine-tuning GPT-2

```python
from transformers import GPT2LMHeadModel, AdamW
from tokenizer import Tokenizer
import torch

# Load GPT-2 tokenizer
tokenizer = Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# Prepare dataset
texts = [
    "In this tutorial, we will learn how to",
    "The quick brown fox jumps over"
]

# Tokenize inputs
inputs = tokenizer.encode_batch(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

# Shift inputs for language modeling
input_ids = inputs["input_ids"]
labels = input_ids.clone()

# Create DataLoader
dataset = torch.utils.data.TensorDataset(input_ids, labels)
dataloader = DataLoader(dataset, batch_size=2)

# Load GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))  # Update for new tokens

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
for epoch in range(3):
    for batch in dataloader:
        batch_input_ids, batch_labels = batch
        
        outputs = model(
            input_ids=batch_input_ids,
            labels=batch_labels
        )
        
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save the model
model.save_pretrained("models/finetuned_gpt2")
tokenizer.save("models/finetuned_gpt2/tokenizer.model")
```

## Custom Dataset with DataLoader

```python
from torch.utils.data import Dataset, DataLoader
from tokenizer import Tokenizer
import torch

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Example usage
texts = ["This is a positive review.", "This is a negative review."]
labels = [1, 0]

tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
dataset = TextClassificationDataset(texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Example training loop
model = ...  # Your model here
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(3):
    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"Loss: {loss.item()}")
```

## Training Loop with PyTorch Lightning

```python
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
from tokenizer import Tokenizer

class TextClassifier(pl.LightningModule):
    def __init__(self, model_name="bert-base-uncased", num_labels=2, learning_rate=2e-5):
        super().__init__()
        self.save_hyperparameters()
        
        # Load pre-trained model
        self.model = ...  # Your model here
        self.tokenizer = Tokenizer.from_pretrained(model_name)
        
        # Metrics
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
    
    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['label']
        )
        
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)
        acc = self.train_acc(preds, batch['label'])
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['label']
        )
        
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)
        acc = self.val_acc(preds, batch['label'])
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.learning_rate)

# Example usage
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
model = TextClassifier()

# Prepare data loaders
train_dataset = ...  # Your dataset here
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Train the model
trainer = pl.Trainer(
    max_epochs=3,
    gpus=1 if torch.cuda.is_available() else 0,
    progress_bar_refresh_rate=10
)

trainer.fit(model, train_loader, val_loader)
```

## Saving and Loading Fine-tuned Models

### Saving a Fine-tuned Model

```python
from transformers import AutoModel
from tokenizer import Tokenizer

# After training
model = ...  # Your trained model
tokenizer = ...  # Your tokenizer

# Save model and tokenizer
model.save_pretrained("models/my_finetuned_model")
tokenizer.save("models/my_finetuned_model/tokenizer.model")

# Also save the configuration
import json
config = {
    "model_type": "bert",
    "num_labels": 2,
    "id2label": {0: "NEGATIVE", 1: "POSITIVE"},
    "label2id": {"NEGATIVE": 0, "POSITIVE": 1}
}

with open("models/my_finetuned_model/config.json", "w") as f:
    json.dump(config, f)
```

### Loading a Fine-tuned Model

```python
from transformers import AutoModelForSequenceClassification
from tokenizer import Tokenizer

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("models/my_finetuned_model")
tokenizer = Tokenizer.load("models/my_finetuned_model/tokenizer.model")

# Example inference
text = "This is a positive example."
inputs = tokenizer.encode(text, return_tensors="pt")
outputs = model(**inputs)
predictions = torch.softmax(outputs.logits, dim=1)
```

## Next Steps

- [Basic Usage](../guide/basic-usage.md) - Learn the basics of using the tokenizer
- [Advanced Usage](../guide/advanced-usage.md) - Explore advanced features
- [API Reference](../api/tokenizer.md) - Detailed API documentation
