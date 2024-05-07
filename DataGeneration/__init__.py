from .RawTextProcessor import TextProcessor
from .EntityExtractor import EntityExtractor
from .BERTQAGenerator import QuestionAndAnswerGenerator

class DataGeneration:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.entity_extractor = EntityExtractor()
        self.qa_generator = QuestionAndAnswerGenerator()

    def generate_data(self, raw_data_file_path):
        entities = self.entity_extractor.process_text_file(raw_data_file_path)

        bert_contexts = self.text_processor.process_text(raw_data_file_path)

        questions, train_answers = self.qa_generator.generate_questions_and_answers(bert_contexts, entities)

        return bert_contexts, questions, train_answers
