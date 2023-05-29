import spacy

nlp = spacy.load("en_core_web_lg")

def sentence_to_vector(sentence):
    doc = nlp(sentence)
    vector = doc.vector
    return vector

sentences = [
    "This is the first sentence.",
    "Here comes the second one.",
    "And finally, the third sentence."
]

vectors = [sentence_to_vector(sentence) for sentence in sentences]

for sentence, vector in zip(sentences, vectors):
    print("Sentence:", sentence)
    print("Vector:", len(vector))
    print()
