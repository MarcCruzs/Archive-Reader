from .RawTextProcessor import TextProcessor
from .EntityExtractor import EntityExtractor

class BERTContext:
    def __init__(self, text_processor: TextProcessor, entity_extractor: EntityExtractor, window_size: int = 2):
        self.text_processor = text_processor
        self.entity_extractor = entity_extractor
        self.window_size = window_size

    def generate_bert_context(self, file_path):
        # Get cleaned sentences
        cleaned_sentences = self.text_processor.process_text(file_path)

        # Get entities and their positions
        entities = self.entity_extractor.process_text_file(file_path)

        # Generate BERT context for each entity
        bert_contexts = []
        for entity, _ in entities:
            entity_sentence_index = None
            for i, sentence in enumerate(cleaned_sentences):
                if entity in sentence:
                    entity_sentence_index = i
                    break
            
            if entity_sentence_index is not None:
                start_index = max(0, entity_sentence_index - self.window_size)
                end_index = min(len(cleaned_sentences), entity_sentence_index + self.window_size + 1)
                bert_contexts.extend(cleaned_sentences[start_index:end_index])

        return bert_contexts
