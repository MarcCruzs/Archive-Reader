from .EntityExtractor import EntityExtractor

class QuestionGenerator:
    def __init__(self):
        self.entity_extractor = EntityExtractor()

    def generate_questions(self, file_path):
        entities = self.entity_extractor.process_text_file(file_path)

        questions = {}
        for entity, label in entities:
            if label == "PERSON":
                questions[entity] = [
                    f"What is {entity} known for?",
                    f"What are the contributions of {entity}?",
                    f"Can you provide some information about {entity}?",
                    f"Tell me about {entity}."
                ]
            else:
                questions[entity] = [
                    f"What is {entity}?",
                    f"Can you provide details about {entity}?",
                    f"Explain the significance of {entity}.",
                    f"What are the properties of {entity}?"
                ]

        return questions
