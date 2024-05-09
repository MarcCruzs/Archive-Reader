import torch
from transformers import BertForQuestionAnswering, BertTokenizer

# Load the trained model and tokenizer
model_path = "bert_extractive_qa_model"
model = BertForQuestionAnswering.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

def ask_question(context, question):
    # Tokenize input
    inputs = tokenizer(question, context, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    
    # Perform inference
    outputs = model(**inputs)
    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits) + 1

    # Decode the tokens to get the answer
    answer_tokens = inputs['input_ids'][0][start_index:end_index]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer

# Example usage
question = "What is SUAS?"
context = "more consistently suas or student unmanned aerial systems is a yearly competition with a particular mission for 2024 competition the mission is multiple package delivery companies have tasked uas to deliver packages to customers these uas must avoid each other travel to the customer identify potential drop locations and deliver the package to a safe location the competition is broken down into four mission tasks autonomous flight obstacle avoidance object detection classification"

answer = ask_question(context.lower(), question.lower())
print("Answer 1 :", answer)

question = "What is suas competition technical design?"
context = "odlc and air delivery this report focuses on the progress on the implementation of the odlc and obstacle avoidance mission tasks for details of actual methodology please refer to suas competition technical design document for odlc we are required to detect and classify two types of objects standardized objects emergent objects standardized objects are 85 x 11 in size of various shapes alphanumerics color of shape and color of alphanumeric figure 1 standard object left"
answer = ask_question(context.lower(), question.lower())
print("Answer 2:", answer)


question = "What is ODLC?"
context = "411 return to home and flight termination failsafes 11 412 mission flight boundary11 413 runways for vtol and htol 11 414 flight performance requirements 12 415 ground control station gcs display requirements 1"
answer = ask_question(context.lower(), question.lower())
print("Answer 3:", answer)
