from transformers import pipeline

emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=1)

def detect_emotion(message):
    emotion_result = emotion_classifier(message)[0][0]
    return emotion_result['label'], round(emotion_result['score'], 2)