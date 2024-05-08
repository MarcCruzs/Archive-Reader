import DataGeneration.EntityExtractor as EntityExtractor

# Initialize the EntityExtractor with optional ignore_labels parameter
entity_extractor = EntityExtractor()

# Process a text file and extract named entities
file_path = "raw_datasets/SUAS-Competition-FALL2023-Final-Report.txt"
entities = entity_extractor.process_text_file(file_path)

# Print the extracted named entities
print("Extracted Named Entities:")
for entity, label in entities:
    print(f"{entity}: {label}")