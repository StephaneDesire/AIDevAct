"""Résumé du pipeline load_filter 

1-Charger les tables pull_requests, repositories, users depuis HuggingFace

2-Filtrer les repos avec ≥500 stars

3-Sélectionner PR AI + PR humaines correspondantes

4-Joindre les informations sur l’auteur (type, provider)

5-Calculer :

    Durée review (heures)

    Merge (0/1)

    Nombre de commentaires

    Closed-loop flag

6-Sauvegarder les datasets intermédiaires
"""

# Import de Pandas et lecture des fichiers .parquet depuis HuggingFace
import pandas as pd

pull_requests = pd.read_parquet("hf://datasets/hao-li/AIDev/all_pull_request.parquet")
repositories = pd.read_parquet("hf://datasets/hao-li/AIDev/all_repository.parquet")
users = pd.read_parquet("hf://datasets/hao-li/AIDev/all_user.parquet")

"""
pull_requests : contient toutes les PRs (humaines + AI)
repositories : infos sur chaque repo (stars, langage, etc.)
users : infos sur les auteurs (humains ou agents IA)
"""

# Filtrage des repositories pour ne garder que ceux avec au moins 500 étoiles
repositories_filtered = repositories[repositories['stars'] >= 500]

""" On ne garde que les repos “populaires” pour garantir la qualité et la généralisation"""

# Garder uniquement les PRs associées aux repositories filtrés
pr_filtered = pull_requests[pull_requests['repo_id'].isin(repositories_filtered['id'])].copy()

# Selectionner les pull requests générées par des agents IA + contrôle humain
# AI PR
pr_ai = pr_filtered[pr_filtered['is_ai_generated'] == True].copy()

# PR humaines (même repos, même période) pour comparaison
pr_human = pr_filtered[pr_filtered['is_ai_generated'] == False].copy()

# On peut limiter à la même période que les PR AI pour le contrôle
min_date = pr_ai['created_at'].min()
max_date = pr_ai['created_at'].max()
pr_human = pr_human[(pr_human['created_at'] >= min_date) & (pr_human['created_at'] <= max_date)].copy()

# Joindre les infos sur les utilisateurs
# Ajouter le type d'auteur (AI agent, humain) et le provider
pr_ai = pr_ai.merge(users[['id', 'user_type', 'provider']], 
                    left_on='user_id', right_on='id', how='left')
pr_human = pr_human.merge(users[['id', 'user_type']], 
                           left_on='user_id', right_on='id', how='left')

"""On sait qui a écrit la PR et son type

Pour RQ3 (closed-loop), on aura besoin du provider IA"""

# Conversion des dates en datetime
pr_ai['created_at'] = pd.to_datetime(pr_ai['created_at'], errors='coerce')
pr_ai['closed_at'] = pd.to_datetime(pr_ai['closed_at'], errors='coerce')

pr_human['created_at'] = pd.to_datetime(pr_human['created_at'], errors='coerce')
pr_human['closed_at'] = pd.to_datetime(pr_human['closed_at'], errors='coerce')

# Durée de review (RQ1) (différence entre created_at et closed_at en heures)
pr_ai.loc[:, 'review_duration_hours'] = ((pr_ai['closed_at'] - pr_ai['created_at']).dt.total_seconds() / 3600).fillna(0)
pr_human.loc[:, 'review_duration_hours'] = ((pr_human['closed_at'] - pr_human['created_at']).dt.total_seconds() / 3600).fillna(0)

# Merge/acceptation (RQ1)
pr_ai['merged'] = pr_ai['merged'].astype(int)
pr_human['merged'] = pr_human['merged'].astype(int)

# Nombre de commentaires / commits (RQ2 simplifié)
# Vérifier si les colonnes existent sinon les créer
for col in ['num_comments', 'num_review_comments', 'num_commits_after_review']:
    if col not in pr_ai.columns:
        pr_ai[col] = 0

# Closed-loop flag (RQ3) - placeholder car review_provider absent
pr_ai['closed_loop'] = 0  # 1 si l’agent IA et le bot de review viennent du même fournisseur, 0 sinon

"""Permet de mesurer le biais “closed-loop”"""

# Sauvegarder le dataset intermédiaire
pr_ai.to_parquet("data/intermediate/pr_ai_filtered.parquet", index=False)
pr_human.to_parquet("data/intermediate/pr_human_filtered.parquet", index=False)
"""Datasets intermédiaires sauvegardés pour analyse ultérieure"""
