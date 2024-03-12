from haystack.retriever import PDFToTextConverter

filter = {"group":["SUAS","SAR"]}

pdf_converter = PDFToTextConverter(remove_numeric_tables=True, valid_languages=["en"])

converted = pdf_converter.convert(file_path = "raw_datasets/SUAS-Competition-FALL-2023-Final-Report.pdf", meta = { "group": "SUAS", "processed": False })


from haystack.nodes import PreProcessor

preprocessor = PreProcessor(split_by="word",
                            split_length=200,
                            split_overlap=10)

preprocessed = preprocessor.process(converted)

from haystack.document_stores import ElasticsearchDocumentStore

document_store = ElasticsearchDocumentStore()
document_store.delete_all_documents()
document_store.write_documents(preprocessed)

from haystack.nodes import DensePassageRetriever, FARMReader

retriever = DensePassageRetriever(document_store=document_store)

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)


document_store.update_embeddings(retriever)


from haystack.pipelines import ExtractiveQAPipeline

pipeline = ExtractiveQAPipeline(reader, retriever)


questions = ["What is the goal of the project?",
             "What does ODLC mean?"]

answers = []

for question in questions:
    prediction = pipeline.run(query=question,
                 params = {"Retriever": {"top_k": 50},
                           "Reader": { "top_k": 1 } })
    answers.append(prediction)

for answer in answers:
    print("Q:", answer["query"])
    print("A:", answer["answers"][0].answer)
    print("\n")