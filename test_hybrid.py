
import numpy as np
from sentence_transformers import SentenceTransformer
from retrieval import build_bm25_index, hybrid_search

documents = [
    {
        "source": "procedure.pdf",
        "page": 1,
        "text": "Le formulaire RF-204 doit être transmis au service financier."
    },
    {
        "source": "procedure.pdf",
        "page": 2,
        "text": "Les congés annuels sont demandés auprès des ressources humaines."
    },
    {
        "source": "procedure.pdf",
        "page": 3,
        "text": "Toute demande de remboursement doit comporter une facture valide."
    }
]

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

embeddings = model.encode(
    [doc["text"] for doc in documents],
    convert_to_numpy=True,
    normalize_embeddings=True
)

bm25_index = build_bm25_index(documents)

questions = [
    "Où envoyer le formulaire RF-204 ?",
    "Comment demander des congés annuels ?",
    "Quel justificatif faut-il pour un remboursement ?"
]

for question in questions:
    results = hybrid_search(
        question=question,
        documents=documents,
        embeddings=embeddings,
        embedding_model=model,
        bm25_index=bm25_index,
        top_k=2,
        candidate_k=3
    )

    print(f"\nQuestion : {question}")

    for result in results:
        print(
            f"Page {result['page']} | "
            f"BM25={result['bm25_score']:.3f} | "
            f"Cosinus={result['semantic_score']:.3f}"
        )
        print(result["text"])
