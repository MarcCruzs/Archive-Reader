from fuzzywuzzy import fuzz

def load_ner_entities(file_path):
    entities = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                # Extracting entity and type from the line
                entity = line.split(': ')[1].split(',')[0].strip()
                type_value = line.split(': ')[2].strip()
                # Ignore certain entity types like Cardinal, Quantity, and Money
                if type_value not in ["CARDINAL", "QUANTITY", "MONEY"]:
                    entities.append((entity, type_value)) 
    return entities

def bio_tagging(ner_entities):
    combined_entities = []
    
    for ner_entity, ner_label in ner_entities:
        # Resetting parameters for a new loop
        found = False
        
        if found:
            # Assuming that all entities found by NER are relevant
            combined_entities.append((ner_entity, "B-" + ner_label))
        else:
            combined_entities.append((ner_entity, "Not Found"))
    return combined_entities

def write_biotagged_output(biotagged_entities, output_file):
    with open(output_file, 'w') as file:
        for entity, label in biotagged_entities:
            file.write(f"{entity}\t{label}\n")

ner_file = "cleaned_datasets/NER_output_entities.txt"
output_file = "cleaned_datasets/bio_annotations.txt"

ner_entities = load_ner_entities(ner_file)
write_biotagged_output(bio_tagging(ner_entities), output_file)
