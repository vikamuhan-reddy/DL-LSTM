# DL- Developing a Deep Learning Model for NER using LSTM

## AIM

To develop an LSTM-based model for recognizing named entities in text using a Bidirectional LSTM (BiLSTM) neural network.

---

# Problem Statement and Dataset

Named Entity Recognition (NER) is a Natural Language Processing (NLP) task used to identify and classify entities such as person names, organizations, locations, dates, etc., from textual data.

Dataset used:

- `ner_dataset.csv`
- Contains:
  - Sentence numbers
  - Words
  - POS tags
  - Named Entity tags


# DESIGN STEPS

### STEP 1:

Import all required libraries and load the NER dataset.

### STEP 2:

Perform preprocessing by handling missing values and creating word/tag vocabularies.

### STEP 3:

Convert words and tags into numerical sequences and pad them to equal length.

### STEP 4:

Create PyTorch Dataset and DataLoader for batching the data.

### STEP 5:

Build the BiLSTM model using Embedding, Dropout, LSTM, and Fully Connected layers.

### STEP 6:

Train the model, evaluate its performance, and visualize the loss curves.

---

# PROGRAM

### Name:

### Register Number:

```python
class BiLSTMTagger(nn.Module):

    def __init__(self, vocab_size, tagset_size,
                 embedding_dim=50, hidden_dim=100):

        super(BiLSTMTagger, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.dropout = nn.Dropout(0.1)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, input_ids):

        x = self.embedding(input_ids)

        x = self.dropout(x)

        x, _ = self.lstm(x)

        x = self.fc(x)

        return x


model = BiLSTMTagger(
    len(word2idx) + 1,
    len(tag2idx)
).to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# Training and Evaluation Functions
def train_model(model, train_loader, test_loader,
                loss_fn, optimizer, epochs=3):

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        # Training Phase
        model.train()

        total_loss = 0.0

        for batch in train_loader:

            input_ids = batch["input_ids"].to(device)

            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids)

            loss = loss_fn(
                outputs.view(-1, len(tag2idx)),
                labels.view(-1)
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        train_losses.append(total_loss)

        # Validation Phase
        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for batch in test_loader:

                input_ids = batch["input_ids"].to(device)

                labels = batch["labels"].to(device)

                outputs = model(input_ids)

                loss = loss_fn(
                    outputs.view(-1, len(tag2idx)),
                    labels.view(-1)
                )

                val_loss += loss.item()

        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {total_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}")

    return train_losses, val_losses
```

---

# OUTPUT

## Loss Vs Epoch Plot
<img width="747" height="581" alt="image" src="https://github.com/user-attachments/assets/0c5a9588-f5d9-4476-9c7d-2122e0118e5b" />


# Sample Text Prediction
<img width="418" height="472" alt="image" src="https://github.com/user-attachments/assets/3228d101-475f-4c6e-8252-cd00266e1f40" />



# RESULT

Thus, a Bidirectional LSTM-based Named Entity Recognition (NER) model was successfully developed using PyTorch. The model effectively identified named entities such as persons, locations, and organizations from text sequences with good prediction accuracy.
