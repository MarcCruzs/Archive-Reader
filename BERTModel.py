from transformers import BertTokenizer, BertForQuestionAnswering
import torch
from torch.utils.data import DataLoader, TensorDataset
from DataGeneration import DataGeneration  # Importing the DataGeneration package

# Load tokenizer and pre-trained BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

# Instantiate DataGeneration class
data_generator = DataGeneration()

# Generate data
raw_data_file = "raw_datasets\SUAS-Competition-FALL2023-Final-Report.txt"
train_contexts, train_questions, train_answers = data_generator.generate_data(raw_data_file)

# Tokenize inputs
inputs = tokenizer(train_contexts, 
                   train_questions, 
                   padding=True, 
                   truncation=True, 
                   return_tensors='pt',
                   max_length=512, 
                   add_special_tokens=True)

# Convert answers to answer indices
start_positions = []
end_positions = []
for answer in train_answers:
    start_positions.append(inputs.char_to_token(answer['start_index']))
    end_positions.append(inputs.char_to_token(answer['end_index'] - 1))

# Convert to PyTorch tensors
start_positions = torch.tensor(start_positions)
end_positions = torch.tensor(end_positions)

# Create a DataLoader
dataset = TensorDataset(inputs.input_ids, inputs.token_type_ids, inputs.attention_mask, start_positions, end_positions)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, token_type_ids, attention_mask, start_positions, end_positions = batch
        outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                        start_positions=start_positions, end_positions=end_positions)
        loss = loss_fn(outputs.start_logits, start_positions) + loss_fn(outputs.end_logits, end_positions)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {total_loss}')

# Save the trained model
model.save_pretrained('bert_qa_model')
