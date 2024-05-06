import spacy

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

def ner_on_text_file(file_path, output_file):
    # Read the text from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Process the text with the NER model
    doc = nlp(text)
    
    # Open the output file to write the named entities
    with open(output_file, 'w', encoding='utf-8') as out_file:
        # Write the named entities to the output file
        for ent in doc.ents:
            # Ignore certain entity types like Cardinal, Quantity, and Money
            if ent.label_ not in ["CARDINAL", "QUANTITY", "MONEY"]:
                out_file.write(f"Entity: {ent.text}, Type: {ent.label_}\n")
                print(f"Entity: {ent.text}, Type: {ent.label_}\n")

file_path = "cleaned_datasets/cleaned_SUAS_final_report.txt"
output_file_path = "cleaned_datasets/NER_output_entities.txt"
ner_on_text_file(file_path, output_file_path)
