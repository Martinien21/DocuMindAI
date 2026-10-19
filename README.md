# DocuMind AI

**Un assistant intelligent pour interroger des documents PDF, basé sur la génération augmentée par récupération (RAG).**

DocuMind AI est une application d'intelligence artificielle qui permet aux utilisateurs de poser des questions sur des documents PDF et d'obtenir des réponses fondées sur les passages pertinents extraits de ces documents.

L'application combine la recherche sémantique, la recherche par mots-clés et un modèle de langage exécuté localement afin de proposer une expérience de recherche documentaire intelligente.

## Fonctionnalités

- **Extraction de documents PDF :** extraction du texte des fichiers PDF à l'aide de PyMuPDF.
- **Découpage intelligent du texte :** segmentation du contenu en passages de taille adaptée, en préservant autant que possible les limites des phrases.
- **Recherche sémantique :** utilisation d'embeddings multilingues pour retrouver les passages proches du sens de la question.
- **Recherche hybride :** combinaison de la recherche sémantique et de la recherche lexicale BM25 à l'aide de la fusion des rangs réciproques (RRF).
- **Génération locale de réponses :** utilisation du modèle Llama 3.2 via Ollama pour générer des réponses à partir du contexte récupéré.
- **Traçabilité des sources :** affichage des noms des documents et des numéros de page associés aux passages retrouvés.
- **Interface interactive :** importation de PDF, questions en langage naturel et consultation des passages récupérés grâce à Streamlit.

## Architecture

Le fonctionnement de DocuMind AI repose sur six étapes principales :

1. **Importation des documents :** les fichiers PDF sont chargés et leur texte est extrait.
2. **Découpage du texte :** le contenu extrait est divisé en passages plus faciles à rechercher et à exploiter.
3. **Indexation :** les passages sont représentés par des embeddings multilingues et indexés pour la recherche lexicale.
4. **Recherche des passages pertinents :** les résultats de la recherche sémantique et de la recherche lexicale sont combinés pour identifier les passages les plus utiles.
5. **Génération de la réponse :** la question de l'utilisateur et les passages récupérés sont transmis au modèle de langage local.
6. **Présentation des résultats :** la réponse générée et les passages justificatifs sont affichés dans l'interface Streamlit.

## Technologies utilisées

- **Python** : langage de programmation principal.
- **Streamlit** : création de l'interface utilisateur.
- **PyMuPDF** : extraction du texte des PDF.
- **Sentence Transformers** : génération des représentations vectorielles des textes.
- **NumPy** : calculs numériques et similarité vectorielle.
- **BM25 (`rank-bm25`)** : recherche lexicale par mots-clés.
- **Ollama** : exécution locale du modèle de langage.
- **Llama 3.2 (`llama3.2:1b`)** : génération des réponses.
- **RAG (Retrieval-Augmented Generation)** : architecture combinant recherche documentaire et génération de texte.

## Prérequis

Avant d'utiliser l'application, il faut disposer des éléments suivants :

- Python installé sur l'ordinateur.
- Ollama installé et opérationnel.
- Un espace disque et une mémoire suffisants pour le modèle d'embeddings et le modèle de langage.

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/Martinien21/documind-ai.git
cd documind-ai
```

### 2. Créer un environnement virtuel

Sous Windows, avec PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Installer les dépendances

```bash
python -m pip install -r requirements.txt
```

### 4. Télécharger le modèle de langage

```bash
ollama pull llama3.2:1b
```

Le téléchargement initial nécessite une connexion Internet. Une fois le modèle installé, l'inférence peut être exécutée localement.

### 5. Lancer l'application

Vérifie qu'Ollama fonctionne, puis exécute :

```bash
python -m streamlit run app.py
```

Ouvre ensuite l'adresse locale affichée dans le terminal pour accéder à l'interface de DocuMind AI.

## Utilisation

1. Lancer l'application DocuMind AI.
2. Importer un ou plusieurs documents PDF.
3. Construire l'index documentaire.
4. Poser une question sur le contenu des documents.
5. Consulter la réponse générée ainsi que les passages récupérés pour la justifier.

## Évaluation

Le projet comprend des tests de base portant sur le découpage du texte et la recherche hybride.

Une prochaine étape consiste à constituer un jeu de questions annotées manuellement afin de comparer les performances de la recherche sémantique, de BM25 et de la recherche hybride.

Des métriques telles que le **Recall@k** et le **MRR (Mean Reciprocal Rank)** permettront de mesurer la capacité du système à retrouver les passages pertinents et leur position dans les résultats.

La qualité des réponses générées, la fiabilité des références aux sources et la capacité du modèle à reconnaître l'absence d'une information devront également être évaluées.

## Limites

- La qualité des réponses dépend de l'extraction du texte, de la pertinence des passages récupérés et des capacités du modèle de langage.
- Les PDF numérisés sous forme d'images peuvent nécessiter une reconnaissance optique de caractères (OCR).
- Un modèle de langage local de petite taille peut produire des réponses incomplètes ou incorrectes.
- Les passages récupérés et les références aux pages doivent être vérifiés à partir des documents originaux.
- Le téléchargement initial des modèles nécessite une connexion Internet, même si leur utilisation peut ensuite s'effectuer localement.

## Confidentialité

DocuMind AI est conçu pour traiter les documents et générer les réponses localement. Les documents confidentiels ne doivent pas être publiés dans un dépôt public.

Avant d'utiliser l'application avec des données sensibles, il convient de vérifier le fonctionnement des services et des dépendances utilisés.

## Feuille de route

- [ ] Évaluer la qualité de la recherche sur un jeu de questions annotées.
- [ ] Comparer les performances de la recherche sémantique, de BM25 et de la recherche hybride.
- [ ] Améliorer la complétude des réponses et la gestion des questions sans réponse dans les documents.
- [ ] Ajouter des tests automatisés pour la recherche et la génération.
- [ ] Améliorer l'interface utilisateur et l'affichage des références documentaires.

## Licence

Aucune licence n'a encore été définie pour ce projet. Les conditions de réutilisation et de redistribution dépendront de la licence qui sera choisie et ajoutée au dépôt.
