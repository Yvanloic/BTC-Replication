# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 01:38:14 2024

@author: 2103020
"""


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Liste des tickers des actions et Bitcoin
tickers = ['SICLQ', 'TSLA', 'BKKT', 'NVDA', 'PYPL', 'SQ', 'COIN', 'BYON', 'MARA', 'MSTR', 'BTC-USD']

# Télécharger les données des prix ajustés (Adj Close) 
data = yf.download(tickers, start="2021-08-06", end="2024-12-13")['Adj Close']

# Nettoyage des données : Vérifier les valeurs manquantes
missing_data = data.isnull().sum()
data_cleaned = data.interpolate(method='linear')  # Imputation des valeurs manquantes

# Transformation des prix en rendements logarithmiques
log_returns = np.log(data_cleaned / data_cleaned.shift(1)).dropna()

# Résumé statistique des rendements
log_returns_summary = log_returns.describe()

# Visualisation des séries temporelles des rendements

colors = cm.tab10.colors  # Utilise une palette de 10 couleurs (ajustable)
for i, ticker in enumerate(log_returns.columns):
    plt.figure(figsize=(10, 5))
    plt.plot(log_returns.index, log_returns[ticker], color=colors[i % len(colors)], label=ticker)  # Couleur unique
    plt.title(f"Série temporelle des rendements logarithmiques - {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Rendement logarithmique")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Histogrammes des rendements
log_returns.hist(bins=50, figsize=(14, 8), grid=False, edgecolor='black')
plt.suptitle("Histogrammes des rendements logarithmiques", fontsize=16)
plt.tight_layout()
plt.show()

# Box-plots des rendements
plt.figure(figsize=(14, 7))
sns.boxplot(data=log_returns)
plt.title("Box-plot des rendements logarithmiques")
plt.xlabel("Actifs")
plt.ylabel("Rendement logarithmique")
plt.show()

# Résumé des rendements
print("Résumé statistique des rendements logarithmiques :")
print(log_returns_summary)

# Exclure Bitcoin des données de rendements journaliers
daily_returns = data_cleaned.pct_change().dropna()
daily_returns_no_bitcoin = daily_returns.drop(columns=["BTC-USD"])  # Supprime la colonne BTC-USD

# Construire un portefeuille également pondéré (uniquement les 10 actions)
num_assets = len(daily_returns_no_bitcoin.columns)  # Nombre d'actifs (sans Bitcoin)
equal_weights = [1 / num_assets] * num_assets  # Poids égaux pour chaque actif

# Rendements du portefeuille : somme pondérée des rendements journaliers
portfolio_returns = daily_returns_no_bitcoin.dot(equal_weights)  # Multiplication matricielle pour obtenir les rendements pondérés

# Valeur cumulée du portefeuille
portfolio_cumulative = (1 + portfolio_returns).cumprod()

# Affichage des résultats
print("Aperçu des rendements journaliers du portefeuille (sans Bitcoin) :")
print(portfolio_returns.head())

print("\nAperçu de la valeur cumulée du portefeuille (sans Bitcoin) :")
print(portfolio_cumulative.head())

# Visualisation des performances du portefeuille
plt.figure(figsize=(10, 5))
plt.plot(portfolio_cumulative.index, portfolio_cumulative, label="Portefeuille Également Pondéré (sans Bitcoin)", color="blue")
plt.title("Valeur cumulée du portefeuille (sans Bitcoin)")
plt.xlabel("Date")
plt.ylabel("Valeur cumulée")
plt.legend()
plt.tight_layout()
plt.show()

# Rebalancement mensuel du portefeuille
def rebalance_portfolio(daily_returns, weights):
    """
    Implémente un rebalancement mensuel du portefeuille également pondéré.
    """
    portfolio_values = [1]  # Initialisation de la valeur du portefeuille à 1
    last_rebalance = daily_returns.index[0]  # Date de rebalancement initiale
    
    for date, row in daily_returns.iterrows():
        # Vérifie si le mois a changé
        if date.month != last_rebalance.month:
            last_rebalance = date  # Mise à jour de la date de rebalancement
            current_value = portfolio_values[-1]
            portfolio_values.append(current_value * (1 + row @ weights))  # Rebalance
        else:
            current_value = portfolio_values[-1]
            portfolio_values.append(current_value * (1 + row @ weights))
    
    return pd.Series(portfolio_values[1:], index=daily_returns.index)

# Supprimer Bitcoin des rendements journaliers
daily_returns_no_bitcoin = daily_returns.drop(columns=["BTC-USD"])
num_assets = len(daily_returns_no_bitcoin.columns)
equal_weights = [1 / num_assets] * num_assets

# Implémenter le rebalancement mensuel
portfolio_cumulative_rebalanced = rebalance_portfolio(daily_returns_no_bitcoin, equal_weights)

# Performance de Bitcoin pour comparaison
bitcoin_cumulative = (1 + daily_returns["BTC-USD"]).cumprod()

# Calcul des métriques de performance
def calculate_metrics(cumulative_returns, daily_returns, benchmark_returns=None):
    """
    Calcule les métriques de performance : Total Return, Volatility, Sharpe Ratio,
    Sortino Ratio, Maximum Drawdown, Calmar Ratio, Tracking Error, Corrélation.
    """
    total_return = cumulative_returns[-1] - 1  # Rendement total
    volatility = daily_returns.std() * np.sqrt(252)  # Annualisation de la volatilité
    sharpe_ratio = (daily_returns.mean() * 252) / volatility  # Ratio de Sharpe annualisé
    
    # Sortino Ratio
    downside_returns = daily_returns[daily_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (daily_returns.mean() * 252) / downside_deviation
    
    # Maximum Drawdown
    rolling_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / rolling_max - 1
    max_drawdown = drawdown.min()
    
    # Calmar Ratio
    calmar_ratio = (cumulative_returns[-1] ** (252 / len(daily_returns)) - 1) / abs(max_drawdown)
    
    # Tracking Error et Corrélation
    tracking_error = None
    correlation = None
    if benchmark_returns is not None:
        tracking_error = np.std(daily_returns - benchmark_returns) * np.sqrt(252)
        correlation = daily_returns.corr(benchmark_returns)
    
    return {
        "Total Return": total_return,
        "Volatility (Ann.)": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Max Drawdown": max_drawdown,
        "Calmar Ratio": calmar_ratio,
        "Tracking Error": tracking_error,
        "Correlation": correlation
    }

# Calcul des métriques pour le portefeuille
portfolio_daily_returns = portfolio_cumulative_rebalanced.pct_change().dropna()
portfolio_metrics = calculate_metrics(portfolio_cumulative_rebalanced, portfolio_daily_returns, daily_returns["BTC-USD"])

# Calcul des métriques pour Bitcoin
bitcoin_daily_returns = daily_returns["BTC-USD"].dropna()
bitcoin_metrics = calculate_metrics(bitcoin_cumulative, bitcoin_daily_returns)

# Affichage des métriques
print("Métriques de performance du portefeuille rebalancé :")
for metric, value in portfolio_metrics.items():
    if value is not None:
        print(f"{metric}: {value:.2%}" if "Return" in metric or "Volatility" in metric or "Drawdown" in metric else f"{metric}: {value:.2f}")

print("\nMétriques de performance de Bitcoin :")
for metric, value in bitcoin_metrics.items():
    if value is not None:
        print(f"{metric}: {value:.2%}" if "Return" in metric or "Volatility" in metric or "Drawdown" in metric else f"{metric}: {value:.2f}")

# Visualisation de la performance cumulée
plt.figure(figsize=(12, 6))
plt.plot(portfolio_cumulative_rebalanced.index, portfolio_cumulative_rebalanced, label="Portefeuille Également Pondéré (Rebalancé)", color="blue")
plt.plot(bitcoin_cumulative.index, bitcoin_cumulative, label="Bitcoin", color="orange")
plt.title("Valeur Cumulée : Portefeuille vs Bitcoin")
plt.xlabel("Date")
plt.ylabel("Valeur cumulée")
plt.legend()
plt.tight_layout()
plt.show()


from sklearn.linear_model import LinearRegression

# Préparer les données pour la régression
X = daily_returns_no_bitcoin  # Rendements des actions (variables explicatives)
y = daily_returns["BTC-USD"]  # Rendements de Bitcoin (variable cible)

# Implémenter la régression linéaire
model = LinearRegression()
model.fit(X, y)

# Récupérer les pondérations
linear_weights = model.coef_

# Normaliser les pondérations pour qu'elles soient proportionnelles
linear_weights_normalized = linear_weights / np.sum(np.abs(linear_weights))

print("Pondérations déterminées par la régression linéaire :")
for ticker, weight in zip(daily_returns_no_bitcoin.columns, linear_weights_normalized):
    print(f"{ticker}: {weight:.4f}")

# Calculer les rendements du portefeuille répliqué
replicated_portfolio_returns = X.dot(linear_weights_normalized)

# Valeur cumulée du portefeuille répliqué
replicated_portfolio_cumulative = (1 + replicated_portfolio_returns).cumprod()

# Visualisation
plt.figure(figsize=(12, 6))
plt.plot(replicated_portfolio_cumulative.index, replicated_portfolio_cumulative, label="Portefeuille Répliqué (Régression Linéaire)", color="green")
plt.plot(bitcoin_cumulative.index, bitcoin_cumulative, label="Bitcoin", color="orange")
plt.title("Valeur Cumulée : Portefeuille Répliqué (Régression Linéaire) vs Bitcoin")
plt.xlabel("Date")
plt.ylabel("Valeur cumulée")
plt.legend()
plt.tight_layout()
plt.show()


# Régression non linéaire avec Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)
rf_predicted_returns = rf_model.predict(X)
rf_portfolio_cumulative = pd.Series((1 + rf_predicted_returns).cumprod(), index=X.index)

# Régression Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X, y)
gb_predicted_returns = gb_model.predict(X)
gb_portfolio_cumulative = pd.Series((1 + gb_predicted_returns).cumprod(), index=X.index)

# Visualisation des performances cumulées
plt.figure(figsize=(12, 6))
plt.plot(rf_portfolio_cumulative.index, rf_portfolio_cumulative, label="Portefeuille Répliqué (Random Forest)", color="purple")
plt.plot(gb_portfolio_cumulative.index, gb_portfolio_cumulative, label="Portefeuille Répliqué (Gradient Boosting)", color="cyan")
plt.plot(bitcoin_cumulative.index, bitcoin_cumulative, label="Bitcoin", color="orange")
plt.title("Valeur Cumulée : Portefeuilles Répliqués vs Bitcoin")
plt.xlabel("Date")
plt.ylabel("Valeur cumulée")
plt.legend()
plt.tight_layout()
plt.show()


# Créer un "Bitcoin Proxy Index" basé sur les pondérations de la régression linéaire
proxy_index_returns = X.dot(linear_weights_normalized)  # Rendements pondérés
proxy_index_cumulative = (1 + proxy_index_returns).cumprod()  # Valeur cumulée

# Visualisation du Bitcoin Proxy Index
plt.figure(figsize=(12, 6))
plt.plot(proxy_index_cumulative.index, proxy_index_cumulative, label="Bitcoin Proxy Index", color="red")
plt.plot(bitcoin_cumulative.index, bitcoin_cumulative, label="Bitcoin", color="orange")
plt.title("Valeur Cumulée : Bitcoin Proxy Index vs Bitcoin")
plt.xlabel("Date")
plt.ylabel("Valeur cumulée")
plt.legend()
plt.tight_layout()
plt.show()

# Analyse des corrélations entre Bitcoin et les actions
correlations = daily_returns_no_bitcoin.corrwith(daily_returns["BTC-USD"])

# Afficher les corrélations
print("Corrélations entre les rendements de Bitcoin et des actions :")
print(correlations)

# Visualisation des corrélations (barplot)
plt.figure(figsize=(12, 6))
correlations.plot(kind='bar', color='skyblue')
plt.title("Corrélations entre Bitcoin et les actions")
plt.xlabel("Actions")
plt.ylabel("Corrélation")
plt.tight_layout()
plt.show()
