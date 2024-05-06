from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

class BERTTrainAnswers:
    def __init__(self):
        pass
    
    def extract_entity_keywords(self, contexts, entities):
        entity_keywords_list = []
        for context in contexts:
            entity_keywords = []
            for entity, label in entities:
                if label == "PERSON":
                    # For PERSON entities, add all words as keywords
                    entity_keywords.extend(entity.split())
                else:
                    # For other entities, add the entity itself as keyword
                    entity_keywords.append(entity)
            entity_keywords_list.append(entity_keywords)
        return entity_keywords_list

    def generate_train_answers(self, contexts, entities):
        entity_keywords_list = self.extract_entity_keywords(contexts, entities)
        train_answers = []
        for entity_keywords, context in zip(entity_keywords_list, contexts):
            context_lower = context.lower()
            answers = []
            for keyword in entity_keywords:
                start_index = context_lower.find(keyword.lower())
                if start_index != -1:
                    end_index = start_index + len(keyword)
                    answers.append((start_index, end_index))
            if answers:  # Only append if there are answers found
                train_answers.append(answers)
        return train_answers
