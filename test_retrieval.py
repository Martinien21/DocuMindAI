
import numpy as np

documents = [
    {
        "text": "Le remboursement exige une facture valide.",
        "page": 1
    },
    {
        "text": "Les congés annuels sont accordés selon le contrat.",
        "page": 2
    },
    {
        "text": "Le dossier de remboursement est examiné par le service financier.",
        "page": 3
    }
]

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

texts = [doc["text"] for doc in documents]
questions = [
    "Quels justificatifs faut-il pour un remboursement ?",
    "Comment sont organisés les congés annuels ?"
]

document_vectors = model.encode(
    texts,
    normalize_embeddings=True
)

question_vectors = model.encode(
    questions,
    normalize_embeddings=True
)

for question, vector in zip(questions, question_vectors):
    scores = document_vectors @ vector
    order = np.argsort(scores)[::-1]

    print(f"\nQuestion : {question}")

    for index in order:
        print(
            f"Score={scores[index]:.3f} | "
            f"Page={documents[index]['page']} | "
            f"{documents[index]['text']}"
        )
