from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from .EntityExtractor import EntityExtractor

class QuestionAndAnswerGenerator:
    def __init__(self):
        self.entity_extractor = EntityExtractor()

    def generate_questions_and_answers(self, raw_data_file_path):
        entities = self.entity_extractor.process_text_file(raw_data_file_path)
        questions, contexts, train_answers = self.generate_questions_and_answers_for_entities(raw_data_file_path, entities)
        return questions, contexts, train_answers

    def generate_questions_and_answers_for_entities(self, raw_data_file_path, entities):
        train_answers = []
        questions = []
        contexts = []

        for entity, label in entities:
            entity_questions = []
            entity_context = None

            # Generate context for the current entity
            entity_context = self.generate_context_for_entity(raw_data_file_path, entity)

            if entity_context:
                if label == "PERSON":
                    entity_questions.extend([
                        f"What is {entity} known for?",
                        f"What are the contributions of {entity}?",
                        f"Can you provide some information about {entity}?",
                        f"Tell me about {entity}."
                    ])
                else:
                    entity_questions.extend([
                        f"What is {entity}?",
                        f"Can you provide details about {entity}?",
                        f"Explain the significance of {entity}.",
                        f"What are the properties of {entity}?"
                    ])

                questions.append(entity_questions)
                contexts.append(entity_context)

                # Generate train answer for the current context
                train_answers.append(self.generate_train_answers(entity_context))

        return questions, contexts, train_answers

    def generate_train_answers(self, context):
        # Tokenize the context
        tokens = word_tokenize(context.lower())

        # Remove stopwords and punctuation
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]

        # Calculate TF-IDF scores
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform([" ".join(tokens)])

        # Get TF-IDF scores for each word
        feature_names = tfidf_vectorizer.get_feature_names_out()
        word_scores = {}
        for col in tfidf_matrix.nonzero()[1]:
            word_scores[feature_names[col]] = tfidf_matrix[0, col]

        # Get the top scoring words
        top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:5]  # Adjust number of top words as needed

        # Find answers based on top scoring words related to the entity
        answers = []
        for keyword, _ in top_words:
            start_index = context.lower().find(keyword)
            if start_index != -1:
                end_index = start_index + len(keyword)
                answers.append((start_index, end_index))

        return answers

    def generate_context_for_entity(self, raw_data_file_path, entity):
        # Get cleaned sentences
        cleaned_sentences = self.text_processor.process_text_file(raw_data_file_path)

        # Find sentences containing the entity
        entity_sentences = [sentence for sentence in cleaned_sentences if entity in sentence]

        # Generate context by concatenating entity sentences
        context = " ".join(entity_sentences)

        return context
