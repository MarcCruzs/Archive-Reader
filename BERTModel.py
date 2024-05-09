import json
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from torch.utils.data import Dataset, DataLoader

import warnings

warnings.filterwarnings("ignore", message="Be aware, overflowing tokens are not returned for the setting you have chosen, i.e. sequence pairs with the 'longest_first' truncation strategy. So the returned list will always be empty even if some tokens have been removed.")

# Load tokenizer and pre-trained BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

# Define a custom dataset class
class QADataset(Dataset):
    def __init__(self, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context']
        question = item['question']
        answer = item['answer']
        start_position, end_position = self.find_answer_span(context, answer)
        if start_position is None or end_position is None:
            # Return None for inputs if answer is not found in context
            print(f"Warning: Answer not found in context for question: {question}")
            return None
        inputs = tokenizer(question, context, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        inputs['start_positions'] = torch.tensor(start_position)
        inputs['end_positions'] = torch.tensor(end_position)
        return inputs

    def find_answer_span(self, context, answer):
        # Tokenize context and answer
        context_tokens = tokenizer.tokenize(context)
        answer_tokens = tokenizer.tokenize(answer)

        # Try to find answer tokens in context tokens
        for start_idx in (i for i, token in enumerate(context_tokens) if token == answer_tokens[0]):
            if context_tokens[start_idx:start_idx + len(answer_tokens)] == answer_tokens:
                return start_idx, start_idx + len(answer_tokens) - 1

        # If answer not found, print a warning
        print(f"Warning: Answer not found in context: {answer}")
        return None, None

def collate(batch):
    batch = [data for data in batch if data is not None]  # Filter out None values
    if len(batch) == 0:
        return None
    input_ids = torch.stack([data['input_ids'].squeeze(0) for data in batch], dim=0)
    attention_mask = torch.stack([data['attention_mask'].squeeze(0) for data in batch], dim=0)
    start_positions = torch.stack([data['start_positions'] for data in batch])
    end_positions = torch.stack([data['end_positions'] for data in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'start_positions': start_positions,
        'end_positions': end_positions,
    }

# Create dataset instance
qa_dataset = QADataset("datasetv2.json")

# Define a data loader with custom collate function
data_loader = DataLoader(qa_dataset, batch_size=8, shuffle=True, collate_fn=collate)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        outputs = model(**batch)
        start_logits, end_logits = outputs.start_logits, outputs.end_logits
        loss = loss_fn(start_logits, batch['start_positions']) + loss_fn(end_logits, batch['end_positions'])
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {total_loss}')

# Save the trained model
model.save_pretrained('bert_extractive_qa_model')
tokenizer.save_pretrained('10EPOCH_bert_extractive_qa_model')
print('Model saved successfully!')
print('Tokenizer saved successfully!')