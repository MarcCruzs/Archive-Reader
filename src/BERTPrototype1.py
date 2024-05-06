import torch
from transformers import BertTokenizer, BertForQuestionAnswering

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

def preprocess_text_and_question(text, question_template):
    # Tokenize text and question
    inputs = tokenizer.encode_plus(question_template, text, add_special_tokens=True, return_tensors="pt")
    return inputs

def get_bert_output(input_tensors):
    # Forward pass through the BERT model
    with torch.no_grad():
        outputs = model(**input_tensors)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    return start_scores, end_scores

def postprocess_bert_output(input_tensors, start_scores, end_scores):
    # Convert start and end scores to probabilities
    start_probs = torch.softmax(start_scores, dim=-1)
    end_probs = torch.softmax(end_scores, dim=-1)
    
    # Find the token indices with the highest start and end probabilities
    start_index = torch.argmax(start_probs)
    end_index = torch.argmax(end_probs)

    # Get the input IDs tensor
    input_ids = input_tensors['input_ids']

    # Exclude special tokens ([CLS] and [SEP]) from the answer span
    while input_ids[0][start_index] in [tokenizer.cls_token_id, tokenizer.sep_token_id]:
        start_index += 1
    while input_ids[0][end_index] in [tokenizer.cls_token_id, tokenizer.sep_token_id]:
        end_index -= 1

    # Get the answer span tokens
    answer_tokens = input_ids[0][start_index:end_index+1]

    # Convert answer tokens to text
    answer = tokenizer.decode(answer_tokens)
    
    return answer

# Example cleaned text and list of question templates
text = "The United Nations (UN) is an international organization founded in 1945. Its stated aims are to promote world peace, human rights, and social progress."
question_templates = ["What is the United Nations?", "When was the United Nations founded?", "What are the aims of the United Nations?"]

# Loop through each question template
for question_template in question_templates:
    # Preprocess text and question
    input_tensors = preprocess_text_and_question(text, question_template)

    # Get BERT output
    start_scores, end_scores = get_bert_output(input_tensors)

    # Postprocess BERT output
    answer = postprocess_bert_output(input_tensors, start_scores, end_scores)

    print("Question:", question_template)
    print("Answer:", answer)
    print()
