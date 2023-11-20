from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, TextClassificationPipeline

KOREAN_MODEL_PATH = 'burningfalls/KBERT-sentiment-analysis'
loaded_tokenizer_korean = AutoTokenizer.from_pretrained(KOREAN_MODEL_PATH)
loaded_model_korean = TFAutoModelForSequenceClassification.from_pretrained(KOREAN_MODEL_PATH)

text_classifier_korean = TextClassificationPipeline(
    tokenizer=loaded_tokenizer_korean,
    model=loaded_model_korean,
    framework='tf',
    top_k=1
)


ENGLISH_MODEL_PATH = 'burningfalls/EBERT-sentiment-analysis'
loaded_tokenizer_english = AutoTokenizer.from_pretrained(ENGLISH_MODEL_PATH)
loaded_model_english = TFAutoModelForSequenceClassification.from_pretrained(ENGLISH_MODEL_PATH)

text_classifier_english = TextClassificationPipeline(
    tokenizer=loaded_tokenizer_english,
    model=loaded_model_english,
    framework='tf',
    top_k=1
)