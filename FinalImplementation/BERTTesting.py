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

def retrieve_context(context, question, num_sentences_per_chunk=3):
    # Removing unneccessary spacing to keep the context more readable
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
    print(inputs)
    # Perform inference
    outputs = model(**inputs)
    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits) + 1

    # Decode the tokens to get the answer
    answer_tokens = inputs['input_ids'][0][start_index:end_index]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer

context = """
This is a progress report for the Software team. This document contains current progress of the project, and also adding any new revisions that the 2025 Software Team to consider to increase the odds of the CPP Broncos team to win more consistently. SUAS, or Student Unmanned Aerial Systems, is a yearly competition with a particular mission. For 2024 Competition, the mission is “Multiple package delivery companies have tasked UAS to deliver packages to customers. These UAS must avoid each other, travel to the customer, identify potential drop locations, and deliver the package to a safe location. The competition is broken down into four mission tasks: Autonomous Flight, Obstacle Avoidance, Object Detection, Classification, Localization (ODLC), and Air Delivery. This report focuses on the progress on the implementation of the ODLC and Obstacle Avoidance mission tasks. For details of actual methodology please refer to SUAS Competition - Technical Design DocumentFor ODLC, we are required to detect and classify two types of objects: standardized objects & emergent objects. Standardized objects are 8.5” x 11” in size of various shapes, alphanumeric/s, color of shape, and color of alphanumeric. The potential shapes according to the competition rulebook: circle, semicircle, quarter circle, triangle, rectangle, pentagon, star, and cross The potential colors according to the competition rulebook: white, black, red, blue, green, purple, brown, and orange. These targets are to be detected by the minimum altitude of 75 ft, and ideally within the range of 85 - 90 ft. Then with the classification of the target to be able to determine localization and signaling payload drop and coordinates to UAV.
"""

def print_question_answer(question_number, question):
    print(f"\nQuestion {question_number}")
    best_chunk = retrieve_context(context.lower(), question.lower())
    answer = ask_question(best_chunk, question)
    print("Question:", question)
    print("----------------")
    print("Answer:", answer)

print_question_answer(1, "What is SUAS?")
print_question_answer(2, "What is the SUAS objects?")
print_question_answer(3, "What is ODLC?")
