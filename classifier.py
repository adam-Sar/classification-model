from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from deep_translator import GoogleTranslator
from langdetect import detect
import torch
import torch.nn.functional as F

# Load Model 1 (binary classifier)
model1_path = "./saved_model"
model1_tokenizer = AutoTokenizer.from_pretrained(model1_path)
model1 = AutoModelForSequenceClassification.from_pretrained(model1_path)

# Load Model 2 (zero-shot classification)
model2_name = "microsoft/deberta-v2-xlarge-mnli"
classifier2 = pipeline(task="zero-shot-classification", model=model2_name)

# Labels
danger_labels = ["threatening", "not-threatening"]
detailed_labels = ["war", "natural-disaster", "terrorism", "crime"]
hypothesis_template = "Does this text describe a dangerous situation: {}"

# Translator
translator = GoogleTranslator(source="auto", target="en")

# Helper: Translate if needed
def translate_if_needed(text):
    return translator.translate(text) if detect(text) != "en" else text

# Main classification function
def classify_article(text, threshold=0.6, weight_model1=0.2, weight_model2=0.8):
    # Step 1: Translate
    text = translate_if_needed(text)

    # Step 2: Binary classification from Model 1
    inputs = model1_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model1(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        threat_prob_1 = probs[0][1].item()  # Model 1's threatening probability 

    # Step 3: Zero-shot classification from Model 2 for "threatening" vs "not-threatening"
    result_danger = classifier2(text, candidate_labels=danger_labels)
    threat_prob_2 = result_danger['scores'][result_danger['labels'].index("threatening")]

    # Combine both model probabilities with weights
    combined_threat_prob = (threat_prob_1 * weight_model1) + (threat_prob_2 * weight_model2)

    if combined_threat_prob >= threshold:
        # If combined probability is high enough, do further categorization
        result = classifier2(text, candidate_labels=detailed_labels, hypothesis_template=hypothesis_template)
        scores_diff = result['scores'][0] - result['scores'][1] <= 0.1
        return {
            "threatening": True,
            "confidence_model1": threat_prob_1,
            "confidence_model2": threat_prob_2,
            "combined_confidence": combined_threat_prob,
            "category": result['labels'][0] if not scores_diff else [result['labels'][0], result['labels'][1]],
            "category_confidence": result['scores'][0] if not scores_diff else result['scores'][0:2]
        }

    return {
        "threatening": False,
        "confidence_model1": threat_prob_1,
        "confidence_model2": threat_prob_2,
        "combined_confidence": combined_threat_prob
    }

