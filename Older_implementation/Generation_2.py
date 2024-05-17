import os
import json
import re
import nltk
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load spaCy NER model
nlp = spacy.load("en_core_web_lg")

# Download NLTK resources
nltk.download('punkt')

# Set up stop words
stop_words = set(nltk.corpus.stopwords.words('english'))

def read_text_files(input_folder):
    texts = []
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                texts.append(text)
    return texts

def preprocess_text(text, max_length=512):
    # Lowercase text
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove tabs
    text = text.replace('\t', ' ')
    # Remove trailing spaces
    text = text.strip()
    tokens = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = tokens['input_ids'].flatten()
    attention_mask = tokens['attention_mask'].flatten()

    return input_ids, attention_mask

def extract_answers(text, question, context_window_size=10):
    # Remove "?" from the question
    keyword = question.split()[-1].rstrip("?")

    # Find the start and end indices of the keyword in the text
    start_index = text.lower().find(keyword.lower())
    if start_index != -1:
        # Split the text into words
        words = text.split()

        try:
            # Find the index of the keyword in the list of words
            keyword_index = words.index(keyword.lower())

            # Find the closest word boundaries before and after the keyword
            context_start = max(0, keyword_index - context_window_size)
            context_end = min(len(words), keyword_index + context_window_size + 1)

            # Extract the context window
            answer_words = words[context_start:context_end]
            answer = " ".join(answer_words)

            # Find the exact start and end indices of the answer in the text
            exact_start_index = text.lower().find(answer.lower())
            exact_end_index = exact_start_index + len(answer)
            return answer, exact_start_index, exact_end_index
        except ValueError:
            # Handle the case where the keyword is not found in the list of words
            return None, None, None

    # If keyword not found, return None
    return None, None, None





def generate_questions(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ['ORG', 'PRODUCT', 'PERSON']]  # Filter entities by type

    questions = set()  # Use a set to automatically handle uniqueness

    for entity, label in entities:
        if label == 'PERSON':
            questions.add(f"Who is {entity}?")
        elif label == 'PRODUCT':
            questions.add(f"What is {entity}?")
            questions.add(f"What does {entity} do?")
        elif label == 'ORG':
            questions.add(f"What is {entity}?")
            questions.add(f"What does {entity} do?")
        

    return list(questions)





def create_dataset(texts):
    dataset = []
    for text in texts:
        # Split the text into chunks of 512 tokens
        text_chunks = [text[i:i+512] for i in range(0, len(text), 512)]
        for chunk in text_chunks:
            input_ids, attention_mask = preprocess_text(chunk)
            preprocessed_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids))
            questions = generate_questions(preprocessed_text)
            for question in questions:
                answer, start_index, end_index = extract_answers(preprocessed_text, question)
                if answer is not None:
                    # Check if the answer span is within the context
                    if start_index is not None and end_index is not None and start_index >= 0 and end_index <= len(preprocessed_text):
                        dataset.append({
                            "question": question,
                            "answer": answer,
                            "context": preprocessed_text,
                            "start_index": start_index,
                            "end_index": end_index,
                            "attention_mask": attention_mask.tolist()  # Convert tensor to list
                        })
    return dataset

def save_dataset(dataset, output_filename):
    with open(output_filename, 'w', encoding='utf-8') as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)

# Example Usage
if __name__ == "__main__":
    input_folder = "raw_files/"
    output_filename = "datasetv4.json"

    texts = read_text_files(input_folder)
    dataset = create_dataset(texts)
    save_dataset(dataset, output_filename)
    print(f"Dataset saved to {output_filename}")
