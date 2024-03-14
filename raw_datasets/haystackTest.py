from haystack.nodes import QuestionGenerator

text = """Python is an interpreted, high-level, general-purpose programming language. Created by Guido van Rossum
and first released in 1991, Python's design philosophy emphasizes code
readability with its notable use of significant whitespace."""

qg = QuestionGenerator()
result = qg.generate(text)