import spacy

class EntityExtractor:
    def __init__(self, ignore_labels=["CARDINAL", "QUANTITY", "MONEY","WORK_OF_ART", "LAW"]):
        # Load English tokenizer, tagger, parser, NER and word vectors
        self.nlp = spacy.load("en_core_web_lg")
        self.ignore_labels = ignore_labels
    
    def process_text_file(self, file_path):
        # Read the text from the file
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        doc = self.nlp(text)
        
        # Initialize a list to store named entities
        entities = []
        
        # Collect the named entities
        for ent in doc.ents:
            # Ignore certain entity types like Cardinal, Quantity, and Money
            if ent.label_ not in self.ignore_labels:
                # Append the entity text and its label
                entities.append((ent.text, ent.label_))
        
        return entities
