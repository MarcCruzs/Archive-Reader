import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from langdetect import detect

class TextProcessor:
    def __init__(self):
        # Install NLTK packages if not already installed
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/wordnet')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('wordnet')
            nltk.download('stopwords')

    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text

    def remove_text_punctuation_except_period(self, text):
        cleaned_text = re.sub(r'[^\w\s\.]', '', text)
        return cleaned_text

    def lowercase_text(self, text):
        cleaned_text = text.lower()
        return cleaned_text

    def remove_text_leading_trailing_whitespaces(self, text):
        cleaned_text = re.sub('\s+', ' ', text).strip()
        return cleaned_text

    def remove_non_english(self, text):
        sentences = sent_tokenize(text)
        cleaned_sentences = []
        for sentence in sentences:
            try:
                if detect(sentence) == 'en':
                    cleaned_sentences.append(sentence)
            except:
                pass
        return cleaned_sentences

    def convert_numbers_to_text(self, text):
        number_to_text = {
            0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
            6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten',
            11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen',
            15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen',
            19: 'nineteen', 20: 'twenty'
        }

        words = text.split()

        converted_text = []
        for word in words:
            if word.isdigit():
                number = int(word)
                if number >= 0 and number <= 20:
                    converted_text.append(number_to_text[number])
                else:
                    converted_text.append(word)
            else:
                converted_text.append(word)

        converted_text = ' '.join(converted_text)

        return converted_text

    def remove_stopwords(self, text):
        stop_words = set(stopwords.words('english'))
        word_tokens = nltk.word_tokenize(text)
        filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
        return ' '.join(filtered_text)

    def process_text(self, file_path):
        raw_text_data = self.load_data(file_path)
        sentences = self.remove_text_punctuation_except_period(raw_text_data)
        lowercased_sentences = self.lowercase_text(sentences)
        modified_lowercased_sentences = self.remove_text_leading_trailing_whitespaces(lowercased_sentences)
        modified_lowercased_sentences = self.remove_non_english(modified_lowercased_sentences)
        modified_lowercased_sentences = [self.convert_numbers_to_text(sentence) for sentence in modified_lowercased_sentences]
        cleaned_sentences = [self.remove_stopwords(sentence) for sentence in modified_lowercased_sentences]
        return cleaned_sentences
