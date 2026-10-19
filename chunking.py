import re
# -----------------------------
# Découpage du texte
# Divise le texte en passages de 1 000 caractères, avec un chevauchement de 150 caractères. Cela évite notamment de perdre certaines informations situées à la frontière de deux passages.
# -----------------------------

def split_text(text, chunk_size=1000, overlap=150):
    """
    Découpe le texte en passages en privilégiant
    les frontières de phrases.

    chunk_size : taille maximale souhaitée en caractères
    overlap    : taille approximative du chevauchement
    """
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    # Séparation approximative des phrases.
    # Les documents avec des abréviations particulières
    # nécessiteront éventuellement un traitement spécialisé.
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_sentences = []
    current_length = 0

    def save_chunk(parts):
        chunk = " ".join(parts).strip()
        if chunk:
            chunks.append(chunk)

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        # Si une phrase dépasse à elle seule la taille cible,
        # on la découpe en blocs pour éviter un passage énorme.
        if len(sentence) > chunk_size:
            if current_sentences:
                save_chunk(current_sentences)
                current_sentences = []
                current_length = 0

            for start in range(0, len(sentence), chunk_size - overlap):
                piece = sentence[start:start + chunk_size].strip()
                if piece:
                    chunks.append(piece)

            continue

        additional_length = len(sentence) + (
            1 if current_sentences else 0
        )

        if (
            current_sentences
            and current_length + additional_length > chunk_size
        ):
            save_chunk(current_sentences)

            # Conserver les dernières phrases pour créer
            # un chevauchement entre deux passages.
            overlap_sentences = []
            overlap_length = 0

            for previous in reversed(current_sentences):
                extra = len(previous) + (
                    1 if overlap_sentences else 0
                )

                if overlap_length + extra > overlap:
                    break

                overlap_sentences.insert(0, previous)
                overlap_length += extra

            current_sentences = overlap_sentences
            current_length = sum(
                len(part) for part in current_sentences
            ) + max(0, len(current_sentences) - 1)

        if current_sentences:
            current_length += 1

        current_sentences.append(sentence)
        current_length += len(sentence)

    if current_sentences:
        save_chunk(current_sentences)

    return chunks
