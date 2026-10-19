
import re
from app import split_text

text = """
Le remboursement est soumis à certaines conditions.
Le demandeur doit fournir une facture valide.
Le dossier est ensuite examiné par le service compétent.
Une notification est envoyée après la décision.
"""

chunks = split_text(text, chunk_size=100, overlap=30)

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- Passage {i} ---")
    print(chunk)
    print("Longueur :", len(chunk))

assert len(chunks) > 0
assert all(chunk.strip() for chunk in chunks)

print("\nTest de découpage réussi.")
