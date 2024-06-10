# https://www.nlplanet.org/course-practical-nlp/02-practical-nlp-first-tasks/08-emotion-classification
from transformers import pipeline

pipe = pipeline(
    "text-classification", model="j-hartmann/emotion-english-distilroberta-base"
)
