import spacy

class EntityExtractor:
    def __init__(self):
        # Load English tokenizer, tagger, parser, NER and word vectors
        self.nlp = spacy.load("en_core_web_lg")
    
    def process_text_file(self, file_path):
        # Read the text from the file
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Process the text with the NER model
        doc = self.nlp(text)
        
        # Initialize a list to store named entities
        entities = []
        
        # Collect the named entities
        for ent in doc.ents:
            # Ignore certain entity types like Cardinal, Quantity, and Money
            if ent.label_ not in ["CARDINAL", "QUANTITY", "MONEY"]:
                if ent.label_ == "PERSON":
                    # Split PERSON entity into individual words
                    for token in ent.text.split():
                        entities.append((token, ent.label_))
                else:
                    entities.append((ent.text, ent.label_))
        
        return entities
    