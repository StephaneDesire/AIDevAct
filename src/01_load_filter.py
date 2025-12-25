"""
Résumé du pipeline load_filter avec PRs humaines via GitHub API

1. Charger les tables pull_requests, repositories, users depuis HuggingFace
2. Filtrer les repos avec ≥500 stars
3. Sélectionner PR AI + PR humaines correspondantes
4. Récupérer les PR humaines via GitHub API si absentes
5. Joindre les informations sur l’auteur (type, login)
6. Calculer :
    - Durée review (heures)
    - Merge (0/1)
    - Nombre de commentaires
    - Closed-loop flag
7. Sauvegarder les datasets intermédiaires
"""

import os
import pandas as pd
import requests
from dotenv import load_dotenv
from datetime import datetime

# -----------------------------
# Charger le token GitHub depuis .env
# -----------------------------
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("Le token GitHub n'est pas défini dans le fichier .env !")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# -----------------------------
#  Charger les datasets HuggingFace
# -----------------------------
pull_requests = pd.read_parquet("hf://datasets/hao-li/AIDev/all_pull_request.parquet")
repositories = pd.read_parquet("hf://datasets/hao-li/AIDev/all_repository.parquet")
users = pd.read_parquet("hf://datasets/hao-li/AIDev/all_user.parquet")

# -----------------------------
#  Filtrer les repositories populaires (≥500 stars)
# -----------------------------
repositories_filtered = repositories[repositories['stars'] >= 500]
repos_list = repositories_filtered['full_name'].tolist()  # format owner/repo

# -----------------------------
#  Séparer les PR AI et PR humaines existantes
# -----------------------------
pr_ai = pull_requests[pull_requests['agent'] != 'human'].copy()
pr_human = pull_requests[pull_requests['agent'] == 'human'].copy()

# -----------------------------
#  Limiter les PRs humaines à la période des PR AI
# -----------------------------
pr_ai['created_at'] = pd.to_datetime(pr_ai['created_at'], utc=True)
pr_ai['closed_at'] = pd.to_datetime(pr_ai['closed_at'], utc=True)

if not pr_human.empty:
    pr_human['created_at'] = pd.to_datetime(pr_human['created_at'], utc=True)
    pr_human['closed_at'] = pd.to_datetime(pr_human['closed_at'], utc=True)

min_date = pr_ai['created_at'].min().strftime("%Y-%m-%d")
max_date = pr_ai['created_at'].max().strftime("%Y-%m-%d")

# -----------------------------
#  Fonction pour récupérer PR humaines via GitHub API
# -----------------------------
def fetch_human_prs(owner_repo, start_date=min_date, end_date=max_date):
    """
    Récupère les PR humaines pour un dépôt donné via GitHub API
    """
    prs = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{owner_repo}/pulls"
        params = {
            "state": "all",
            "per_page": 100,
            "page": page,
            "sort": "created",
            "direction": "asc"
        }
        response = requests.get(url, headers=HEADERS, params=params)
        data = response.json()
        if not data:
            break

        for pr in data:
            created_at = pr.get('created_at')
            if start_date <= created_at[:10] <= end_date:
                user_type = pr['user'].get('type', '')
                login = pr['user'].get('login', '')
                # Exclure les bots
                if user_type == "User" and "bot" not in login.lower():
                    prs.append({
                        "id": pr['id'],
                        "repo_full_name": owner_repo,
                        "user_login": login,
                        "user_id": pr['user']['id'],
                        "created_at": pr['created_at'],
                        "closed_at": pr.get('closed_at', None),
                        "merged_at": pr.get('merged_at', None),
                        "num_comments": pr.get('comments', 0),              # KeyError évité
                        "num_review_comments": pr.get('review_comments', 0),
                        "num_commits_after_review": 0,                     # Placeholder
                        "agent": "human"
                    })
        page += 1
    return prs

# -----------------------------
#  Compléter pr_human via GitHub API si vide
# -----------------------------
if pr_human.empty:
    all_human_prs = []
    for repo_full_name in repos_list:
        human_prs = fetch_human_prs(repo_full_name)
        all_human_prs.extend(human_prs)
    pr_human = pd.DataFrame(all_human_prs)
    pr_human['created_at'] = pd.to_datetime(pr_human['created_at'], utc=True)
    pr_human['closed_at'] = pd.to_datetime(pr_human['closed_at'], utc=True)

# -----------------------------
# Joindre les infos utilisateurs
# -----------------------------
pr_ai = pr_ai.merge(users[['id', 'login']], left_on='user_id', right_on='id', how='left')
pr_human = pr_human.merge(users[['id', 'login']], left_on='user_id', right_on='id', how='left')

# -----------------------------
# Calcul des métriques
# -----------------------------
for df in [pr_ai, pr_human]:
    # Durée review (heures)
    df['review_duration_hours'] = (df['closed_at'] - df['created_at']).dt.total_seconds() / 3600
    # Merge (0/1)
    df['merged'] = df['merged_at'].notna().astype(int)
    # Colonnes de commentaires
    for col in ['num_comments', 'num_review_comments', 'num_commits_after_review']:
        if col not in df.columns:
            df[col] = 0

# Closed-loop flag (placeholder)
pr_ai['closed_loop'] = 0

# -----------------------------
# Sauvegarder les datasets intermédiaires
# -----------------------------
os.makedirs("data/intermediate", exist_ok=True)
pr_ai.to_parquet("data/intermediate/pr_ai_filtered.parquet", index=False)
pr_human.to_parquet("data/intermediate/pr_human_filtered.parquet", index=False)
