import torch
from transformers import BertForQuestionAnswering, BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

# Load the trained model and tokenizer
model_path = "./demo_bert_extractive_qa_model_V3"
model = BertForQuestionAnswering.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

def load_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def retrieve_context(context, question, num_sentences_per_chunk=3):
    # Removing unnecessary spacing to keep the context more readable
    context = ' '.join(context.split())
    context = ' '.join(context.split('\n'))

    # Tokenize question and extract keyword
    question_tokens = nltk.word_tokenize(question)
    keyword = question_tokens[-2]  # Assuming the format "What is [Keyword]?"

    # Tokenize context into sentences
    sentences = sent_tokenize(context)

    # Find sentences containing the keyword
    keyword_sentences = [sentence for sentence in sentences if keyword in nltk.word_tokenize(sentence)]

    # If no sentence contains the keyword, select the sentences with the highest TF-IDF score
    if len(keyword_sentences) == 0:
        # Calculate TF-IDF scores for each sentence
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([question + " " + sentence for sentence in sentences])
        question_tfidf = vectorizer.transform([question])

        print("(Sntnce Index)\t(TF-IDF Score)")
        print("(Strt, End)\n", tfidf_matrix)
        

        # Calculate cosine similarity between question TF-IDF and sentence TF-IDF
        similarity_scores = np.dot(tfidf_matrix, question_tfidf.T).toarray().flatten()

        # Select the sentences with the highest similarity scores
        top_sentence_indices = np.argsort(similarity_scores)[::-1][:num_sentences_per_chunk]
        best_chunk = ' '.join([sentences[i] for i in top_sentence_indices])

        return best_chunk

    # If keyword sentences are found, form chunks with multiple sentences
    chunks = []
    for i in range(0, len(keyword_sentences), num_sentences_per_chunk):
        chunk = ' '.join(keyword_sentences[i:i+num_sentences_per_chunk])
        chunks.append(chunk)

    return chunks[0]

def ask_question(context, question):
    # Tokenize input
    inputs = tokenizer(question, context, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    #print(inputs)
    # Perform inference
    outputs = model(**inputs)
    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits) + 1

    # Decode the tokens to get the answer
    answer_tokens = inputs['input_ids'][0][start_index:end_index]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer

# Load the corpus from the file
corpus_path = "SUAS-Competition-FALL2023-Final-Report.txt"
context = load_corpus(corpus_path)

def print_question_answer(question_number, question):
    print(f"\nQuestion {question_number}")
    best_chunk = retrieve_context(context.lower(), question.lower())
    answer = ask_question(best_chunk, question)
    print("Question:", question)
    print("----------------")
    print("Answer:", answer)

# Dynamic question asking
while True:
    user_input = input("Enter your question (or type 'exit' to stop): ")
    if user_input.lower() == 'exit':
        break
    print_question_answer(1, user_input)