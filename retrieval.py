
import re
import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text):
    """
    Tokenisation simple pour BM25.
    Préserve les codes comme RF-204 et les nombres.
    """
    return re.findall(
        r"\w+(?:[-./]\w+)*",
        text.lower(),
        flags=re.UNICODE
    )


def build_bm25_index(documents):
    """
    Construit l'index lexical à partir des passages.
    """
    if not documents:
        raise ValueError("La liste des documents est vide.")

    tokenized_documents = [
        tokenize(doc["text"])
        for doc in documents
    ]

    return BM25Okapi(tokenized_documents)


def reciprocal_rank_fusion(
    rankings,
    k=60,
    weights=None
):
    """
    Fusionne plusieurs classements avec RRF.

    rankings : listes d'indices, du meilleur au moins bon.
    weights  : poids de chaque classement.
    """
    if weights is None:
        weights = [1.0] * len(rankings)

    if len(weights) != len(rankings):
        raise ValueError(
            "Le nombre de poids doit correspondre "
            "au nombre de classements."
        )

    scores = {}

    for ranking, weight in zip(rankings, weights):
        for rank, doc_index in enumerate(ranking, start=1):
            scores[doc_index] = (
                scores.get(doc_index, 0.0)
                + weight / (k + rank)
            )

    return sorted(
        scores,
        key=scores.get,
        reverse=True
    )


def hybrid_search(
    question,
    documents,
    embeddings,
    embedding_model,
    bm25_index,
    top_k=4,
    candidate_k=10,
    semantic_weight=1.0,
    lexical_weight=1.0
):
    """
    Recherche sémantique + BM25 + fusion RRF.

    embeddings : matrice des embeddings des passages,
                 normalisés comme dans build_index().
    """
    if not documents:
        return []

    if len(documents) != len(embeddings):
        raise ValueError(
            "Le nombre de passages et d'embeddings diffère."
        )

    # 1. Classement sémantique
    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    semantic_scores = embeddings @ query_embedding

    semantic_order = np.argsort(
        semantic_scores
    )[::-1][:candidate_k].tolist()

    # 2. Classement lexical BM25
    lexical_scores = np.asarray(
        bm25_index.get_scores(tokenize(question))
    )

    lexical_order = np.argsort(
        lexical_scores
    )[::-1][:candidate_k].tolist()

    # 3. Fusion des classements
    fused_order = reciprocal_rank_fusion(
        [semantic_order, lexical_order],
        weights=[semantic_weight, lexical_weight]
    )

    # 4. Retourner les passages classés
    results = []

    for doc_index in fused_order[:top_k]:
        result = documents[doc_index].copy()
        result["semantic_score"] = float(
            semantic_scores[doc_index]
        )
        result["bm25_score"] = float(
            lexical_scores[doc_index]
        )
        results.append(result)

    return results
