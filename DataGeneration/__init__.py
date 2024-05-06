from .RawTextProcessor import TextProcessor
from .EntityExtractor import EntityExtractor
from .BERTContext import BERTContext
from .BERTQuestionGenerator import QuestionGenerator
from .BERTTrainAnswers import BERTTrainAnswers

class DataGeneration:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.entity_extractor = EntityExtractor()
        self.bert_context_generator = BERTContext(self.text_processor, self.entity_extractor)
        self.question_generator = QuestionGenerator()
        self.train_answer_generator = BERTTrainAnswers()

    def generate_data(self, raw_data_file_path):
        # cleaned_text = self.text_processor.process_text(raw_data_file_path)

        entities = self.entity_extractor.process_text_file(raw_data_file_path)

        bert_contexts = self.bert_context_generator.generate_bert_context(raw_data_file_path)

        questions = self.question_generator.generate_questions(raw_data_file_path)        

        # Generate train answers for each context
        train_answers = self.train_answer_generator.generate_train_answers(bert_contexts, entities)

        return bert_contexts, questions, train_answers


