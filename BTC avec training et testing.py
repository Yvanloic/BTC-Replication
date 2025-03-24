# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 11:35:28 2024

@author: 2103020
"""

# IMPORTATIONS
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from matplotlib import cm

# 1. COLLECTE ET PRÉPARATION DES DONNÉES
    tickers = ['SICLQ', 'TSLA', 'BKKT', 'NVDA', 'PYPL', 'SQ', 'COIN', 'BYON', 'MARA', 'MSTR', 'BTC-USD']
    data = yf.download(tickers, start="2021-08-06", end="2024-12-13")['Adj Close']
    data_cleaned = data.interpolate(method='linear')  # Imputation des valeurs manquantes
    daily_returns = data_cleaned.pct_change().dropna()
    
    # Séparer en training et testing
    train_size = int(0.8 * len(daily_returns))
    train_X = daily_returns.iloc[:train_size].drop(columns=["BTC-USD"])
    test_X = daily_returns.iloc[train_size:].drop(columns=["BTC-USD"])
    train_y = daily_returns["BTC-USD"].iloc[:train_size]
    test_y = daily_returns["BTC-USD"].iloc[train_size:]
    bitcoin_cumulative = (1 + daily_returns["BTC-USD"]).cumprod()
    
    # Tracer les prix historiques avec une échelle logarithmique
    plt.figure(figsize=(15, 8))
    
    # Tracer chaque actif
    for ticker in data_cleaned.columns:
        plt.plot(data_cleaned.index, data_cleaned[ticker], label=ticker)
    
    # Ajouter une échelle logarithmique à l'axe des ordonnées
    plt.yscale("log")
    
    # Ajout de titres et légendes
    plt.title("Prix historiques ajustés pour toutes les actions et Bitcoin (Échelle Logarithmique)", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Prix ajusté (log)", fontsize=12)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=10)  # Légende en dehors du graphique
    plt.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

# 2. VISUALISATIONS DES RENDEMENTS
    colors = cm.tab10.colors  # Palette de couleurs
    for i, ticker in enumerate(daily_returns.columns):
        plt.figure(figsize=(10, 5))
        plt.plot(daily_returns.index, daily_returns[ticker], color=colors[i % len(colors)], label=ticker)
        plt.title(f"Série temporelle des rendements logarithmiques - {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Rendement logarithmique")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # Histogrammes des rendements logarithmiques
    daily_returns.hist(bins=50, figsize=(14, 8), grid=False, edgecolor='black')
    plt.suptitle("Histogrammes des rendements logarithmiques", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Box-plots des rendements logarithmiques
    plt.figure(figsize=(14, 7))
    sns.boxplot(data=daily_returns)
    plt.title("Box-plot des rendements logarithmiques")
    plt.xlabel("Actifs")
    plt.ylabel("Rendement logarithmique")
    plt.tight_layout()
    plt.show()

# 3. PORTFOLIO ÉGALEMENT PONDÉRÉ
    num_assets = len(train_X.columns)
    equal_weights = [1 / num_assets] * num_assets
    
    # Training Portfolio
    train_portfolio_returns = train_X.dot(equal_weights)
    train_portfolio_cumulative = (1 + train_portfolio_returns).cumprod()
    
    # Testing Portfolio
    test_portfolio_returns = test_X.dot(equal_weights)
    test_portfolio_cumulative = (1 + test_portfolio_returns).cumprod()
    
    plt.figure(figsize=(12, 6))
    plt.plot(train_portfolio_cumulative.index, train_portfolio_cumulative, label="Portfolio Également Pondéré (Training)", color="blue")
    plt.plot(test_portfolio_cumulative.index, test_portfolio_cumulative, label="Portfolio Également Pondéré (Testing)", color="cyan")
    plt.plot(bitcoin_cumulative.index, bitcoin_cumulative, label="Bitcoin", color="orange")
    plt.title("Performance Portefeuille Également Pondéré vs Bitcoin")
    plt.xlabel("Date")
    plt.ylabel("Valeur cumulée")
    plt.legend()
    plt.tight_layout()
    plt.show()

# 4. REBALANCEMENT MENSUEL
    def rebalance_portfolio(daily_returns, weights):
        """
        Implémente un rebalancement mensuel du portefeuille.
        """
        portfolio_values = [1]
        last_rebalance = daily_returns.index[0]
        for date, row in daily_returns.iterrows():
            if date.month != last_rebalance.month:
                last_rebalance = date
                current_value = portfolio_values[-1]
                portfolio_values.append(current_value * (1 + row @ weights))
            else:
                current_value = portfolio_values[-1]
                portfolio_values.append(current_value * (1 + row @ weights))
        return pd.Series(portfolio_values[1:], index=daily_returns.index)
    
    # Rebalancement sur le training set
    train_rebalanced = rebalance_portfolio(train_X, equal_weights)
    train_btc_cumulative = (1 + train_y).cumprod()
    
    # Rebalancement sur le testing set
    test_rebalanced = rebalance_portfolio(test_X, equal_weights)
    test_btc_cumulative = (1 + test_y).cumprod()
    
    # Graphique pour le training set
    plt.figure(figsize=(12, 6))
    plt.plot(train_rebalanced.index, train_rebalanced, label="Portfolio Rebalancé (Training)", color="blue")
    plt.plot(train_btc_cumulative.index, train_btc_cumulative, label="Bitcoin (Training)", color="orange")
    plt.title("Performance Portefeuille Rebalancé vs Bitcoin (Training Set)")
    plt.xlabel("Date")
    plt.ylabel("Valeur Cumulée")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Graphique pour le testing set
    plt.figure(figsize=(12, 6))
    plt.plot(test_rebalanced.index, test_rebalanced, label="Portfolio Rebalancé (Testing)", color="cyan")
    plt.plot(test_btc_cumulative.index, test_btc_cumulative, label="Bitcoin (Testing)", color="orange")
    plt.title("Performance Portefeuille Rebalancé vs Bitcoin (Testing Set)")
    plt.xlabel("Date")
    plt.ylabel("Valeur Cumulée")
    plt.legend()
    plt.tight_layout()
    plt.show()


# 5. RÉPLICATION DE BITCOIN
    linear_model = LinearRegression()
    linear_model.fit(train_X, train_y)
    train_predicted_y = linear_model.predict(train_X)
    test_predicted_y = linear_model.predict(test_X)
    
    train_replicated_cumulative = (1 + pd.Series(train_predicted_y, index=train_X.index)).cumprod()
    test_replicated_cumulative = (1 + pd.Series(test_predicted_y, index=test_X.index)).cumprod()
    
    # Graphique pour la régression linéaire
    plt.figure(figsize=(12, 6))
    plt.plot(train_replicated_cumulative.index, train_replicated_cumulative, label="Réplique Bitcoin (Training)", color="green")
    plt.plot(train_btc_cumulative.index, train_btc_cumulative, label="Bitcoin (Training)", color="orange")
    plt.title("Réplique de Bitcoin par Régression Linéaire (Training)")
    plt.xlabel("Date")
    plt.ylabel("Valeur Cumulée")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(test_replicated_cumulative.index, test_replicated_cumulative, label="Réplique Bitcoin (Testing)", color="purple")
    plt.plot(test_btc_cumulative.index, test_btc_cumulative, label="Bitcoin (Testing)", color="orange")
    plt.title("Réplique de Bitcoin par Régression Linéaire (Testing)")
    plt.xlabel("Date")
    plt.ylabel("Valeur Cumulée")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Random Forest et Gradient Boosting
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(train_X, train_y)
    rf_train_cumulative = (1 + pd.Series(rf_model.predict(train_X), index=train_X.index)).cumprod()
    rf_test_cumulative = (1 + pd.Series(rf_model.predict(test_X), index=test_X.index)).cumprod()
    
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(train_X, train_y)
    gb_train_cumulative = (1 + pd.Series(gb_model.predict(train_X), index=train_X.index)).cumprod()
    gb_test_cumulative = (1 + pd.Series(gb_model.predict(test_X), index=test_X.index)).cumprod()
    
    # Graphiques pour Random Forest et Gradient Boosting
    plt.figure(figsize=(12, 6))
    plt.plot(rf_train_cumulative.index, rf_train_cumulative, label="Random Forest (Training)", color="purple")
    plt.plot(gb_train_cumulative.index, gb_train_cumulative, label="Gradient Boosting (Training)", color="blue")
    plt.plot(train_btc_cumulative.index, train_btc_cumulative, label="Bitcoin (Training)", color="orange")
    plt.title("Modèles Avancés vs Bitcoin (Training)")
    plt.xlabel("Date")
    plt.ylabel("Valeur Cumulée")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(rf_test_cumulative.index, rf_test_cumulative, label="Random Forest (Testing)", color="cyan")
    plt.plot(gb_test_cumulative.index, gb_test_cumulative, label="Gradient Boosting (Testing)", color="green")
    plt.plot(test_btc_cumulative.index, test_btc_cumulative, label="Bitcoin (Testing)", color="orange")
    plt.title("Modèles Avancés vs Bitcoin (Testing)")
    plt.xlabel("Date")
    plt.ylabel("Valeur Cumulée")
    plt.legend()
    plt.tight_layout()
    plt.show()

# 6. BITCOIN PROXY INDEX ET CORRELATIONS
    #a) proxy classique
    proxy_weights = train_X.mean() / train_X.mean().sum()
    proxy_train_cumulative = (1 + train_X.dot(proxy_weights)).cumprod()
    proxy_test_cumulative = (1 + test_X.dot(proxy_weights)).cumprod()
    
    # Graphiques pour Bitcoin Proxy Index
    plt.figure(figsize=(12, 6))
    plt.plot(proxy_train_cumulative.index, proxy_train_cumulative, label="Bitcoin Proxy Index Classique (Training)", color="red")
    plt.plot(train_btc_cumulative.index, train_btc_cumulative, label="Bitcoin (Training)", color="orange")
    plt.title("Bitcoin Proxy Index Classique vs Bitcoin (Training)")
    plt.xlabel("Date")
    plt.ylabel("Valeur Cumulée")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(proxy_test_cumulative.index, proxy_test_cumulative, label="Bitcoin Proxy Index Classique (Testing)", color="pink")
    plt.plot(test_btc_cumulative.index, test_btc_cumulative, label="Bitcoin (Testing)", color="orange")
    plt.title("Bitcoin Proxy Index Classique vs Bitcoin (Testing)")
    plt.xlabel("Date")
    plt.ylabel("Valeur Cumulée")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    #b) proxy avec régression linéaire
    from sklearn.linear_model import LinearRegression
    
    # Optimisation des poids pour le proxy (Training Set)
    reg_model = LinearRegression()
    reg_model.fit(train_X, train_y)  # Ajustement des poids pour minimiser l'écart avec Bitcoin
    proxy_weights_optimized = pd.Series(reg_model.coef_, index=train_X.columns)
    
    # Cumul des rendements du portefeuille proxy (Training et Testing)
    proxy_train_cumulative = (1 + train_X.dot(proxy_weights_optimized)).cumprod()
    proxy_test_cumulative = (1 + test_X.dot(proxy_weights_optimized)).cumprod()
    
    # Graphiques pour Bitcoin Proxy Index
    plt.figure(figsize=(12, 6))
    plt.plot(proxy_train_cumulative.index, proxy_train_cumulative, label="Bitcoin Proxy Index (Regression - Training)", color="red")
    plt.plot(train_btc_cumulative.index, train_btc_cumulative, label="Bitcoin (Training)", color="orange")
    plt.title("Bitcoin Proxy Index Regression vs Bitcoin (Training)")
    plt.xlabel("Date")
    plt.ylabel("Valeur Cumulée")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(proxy_test_cumulative.index, proxy_test_cumulative, label="Bitcoin Proxy Index (Regression - Testing)", color="pink")
    plt.plot(test_btc_cumulative.index, test_btc_cumulative, label="Bitcoin (Testing)", color="orange")
    plt.title("Bitcoin Proxy Index Regression vs Bitcoin (Testing)")
    plt.xlabel("Date")
    plt.ylabel("Valeur Cumulée")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    #c) proxy avec objectif de minimiser le TEV 
    from scipy.optimize import minimize
    
    def optimize_proxy_weights(train_X, train_y, method="tracking_error"):
        """
        Optimise les poids pour minimiser la Tracking Error ou maximiser la Correlation.
        
        Parameters:
            train_X (DataFrame): Rendements journaliers des actifs (features).
            train_y (Series): Rendements journaliers de Bitcoin (target).
            method (str): Critère d'optimisation, "tracking_error" ou "correlation".
        
        Returns:
            Series: Poids optimaux pour chaque actif.
        """
        num_assets = train_X.shape[1]
    
        # Fonction objectif
        def objective(weights):
            proxy_returns = train_X.dot(weights)
            if method == "tracking_error":
                # Minimiser la Tracking Error
                return np.std(proxy_returns - train_y)
            elif method == "correlation":
                # Maximiser la correlation (en minimisant son opposé)
                return -proxy_returns.corr(train_y)
            else:
                raise ValueError("Méthode inconnue : choisir 'tracking_error' ou 'correlation'")
    
        # Contraintes : somme des poids = 1
        constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
    
        # Bornes : les poids doivent être compris entre 0 et 1 (long-only)
        bounds = [(0, 1) for _ in range(num_assets)]
    
        # Initialisation des poids (également pondéré)
        initial_weights = np.ones(num_assets) / num_assets
    
        # Optimisation
        result = minimize(objective, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints)
    
        # Retour des poids optimaux
        return pd.Series(result.x, index=train_X.columns)
    
    # Optimisation des poids pour le Proxy Index
    proxy_weights_optimized = optimize_proxy_weights(train_X, train_y, method="tracking_error")  # Ou "correlation"
    
    # Calcul des rendements cumulés du portefeuille proxy (Training et Testing)
    proxy_train_cumulative = (1 + train_X.dot(proxy_weights_optimized)).cumprod()
    proxy_test_cumulative = (1 + test_X.dot(proxy_weights_optimized)).cumprod()
    
    # Graphiques pour Bitcoin Proxy Index optimisé
    plt.figure(figsize=(12, 6))
    plt.plot(proxy_train_cumulative.index, proxy_train_cumulative, label="Bitcoin Proxy Index (TEV - Training)", color="red")
    plt.plot(train_btc_cumulative.index, train_btc_cumulative, label="Bitcoin (Training)", color="orange")
    plt.title("Bitcoin Proxy Index TEV vs Bitcoin (Training)")
    plt.xlabel("Date")
    plt.ylabel("Valeur Cumulée")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(proxy_test_cumulative.index, proxy_test_cumulative, label="Bitcoin Proxy Index (TEV - Testing)", color="pink")
    plt.plot(test_btc_cumulative.index, test_btc_cumulative, label="Bitcoin (Testing)", color="orange")
    plt.title("Bitcoin Proxy Index TEV vs Bitcoin (Testing)")
    plt.xlabel("Date")
    plt.ylabel("Valeur Cumulée")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # ANALYSE DES CORRELATIONS ENTRE LE BTC ET LES AUTRES ACTIONS AFIN DE CREER UN PROXY REDUIT
    
    # Analyse des corrélations entre Bitcoin et les actions
    correlations = daily_returns.drop(columns=["BTC-USD"]).corrwith(daily_returns["BTC-USD"])
    plt.figure(figsize=(12, 6))
    correlations.plot(kind='bar', color='skyblue')
    plt.title("Corrélations entre Bitcoin et les actions")
    plt.xlabel("Actions")
    plt.ylabel("Corrélation")
    plt.tight_layout()
    plt.show()
    
    # Extraire les corrélations uniquement avec Bitcoin
    btc_action_corr = daily_returns.corr()["BTC-USD"].drop("BTC-USD")
    
    # Heatmap des corrélations (Bitcoin et les autres actifs)
    plt.figure(figsize=(12, 6))
    sns.heatmap(btc_action_corr.to_frame().T, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Corrélation de Bitcoin avec les Autres Actifs")
    plt.xlabel("Actifs")
    plt.ylabel("Bitcoin")
    plt.tight_layout()
    plt.show()
    
    # Identifier les actifs les plus corrélés
    top_assets = correlations.abs().sort_values(ascending=False).head(5).index
    print("Actifs les plus corrélés à Bitcoin :")
    print(correlations.loc[top_assets])
    
    
    #d) Proxy actifs correlés
    # Filtrer les actifs les plus corrélés
    top_correlated_stocks = ["MSTR", "MARA", "COIN", "SQ", "NVDA"]
    
    # Réduire les datasets de training et testing aux actions sélectionnées
    train_X_reduced = train_X[top_correlated_stocks]
    test_X_reduced = test_X[top_correlated_stocks]
    
    # Optimisation des poids pour le proxy réduit
    proxy_weights_reduced = optimize_proxy_weights(train_X_reduced, train_y, method="correlation")  # Ou "correlation"
    
    # Calcul des rendements cumulés pour le proxy réduit
    proxy_train_cumulative_reduced = (1 + train_X_reduced.dot(proxy_weights_reduced)).cumprod()
    proxy_test_cumulative_reduced = (1 + test_X_reduced.dot(proxy_weights_reduced)).cumprod()
    
    # Graphiques pour le proxy réduit
    plt.figure(figsize=(12, 6))
    plt.plot(proxy_train_cumulative_reduced.index, proxy_train_cumulative_reduced, label="Proxy Réduit (Training)", color="green")
    plt.plot(train_btc_cumulative.index, train_btc_cumulative, label="Bitcoin (Training)", color="orange")
    plt.title("Proxy Réduit vs Bitcoin (Training)")
    plt.xlabel("Date")
    plt.ylabel("Valeur Cumulée")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(proxy_test_cumulative_reduced.index, proxy_test_cumulative_reduced, label="Proxy Réduit (Testing)", color="purple")
    plt.plot(test_btc_cumulative.index, test_btc_cumulative, label="Bitcoin (Testing)", color="orange")
    plt.title("Proxy Réduit vs Bitcoin (Testing)")
    plt.xlabel("Date")
    plt.ylabel("Valeur Cumulée")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    #e) proxy lambda_corr
    
    from scipy.optimize import minimize
    
    def optimize_proxy_weights_combined(train_X, train_y, lambda_corr=0.6):
        """
        Optimise les poids pour maximiser la corrélation et minimiser la Tracking Error.
        
        Parameters:
            train_X (DataFrame): Rendements journaliers des actifs (features).
            train_y (Series): Rendements journaliers de Bitcoin (target).
            lambda_corr (float): Poids donné à la corrélation dans l'objectif combiné.
        
        Returns:
            Series: Poids optimaux pour chaque actif.
        """
        num_assets = train_X.shape[1]
    
        # Fonction objectif combinée
        def objective(weights):
            proxy_returns = train_X.dot(weights)
            correlation = proxy_returns.corr(train_y)
            tracking_error = np.std(proxy_returns - train_y)
            # Objectif : maximiser corrélation (-correlation) et minimiser Tracking Error
            return -lambda_corr * correlation + (1 - lambda_corr) * tracking_error
    
        # Contraintes : somme des poids = 1
        constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
    
        # Bornes : les poids doivent être compris entre 0 et 1 (long-only)
        bounds = [(0, 1) for _ in range(num_assets)]
    
        # Initialisation des poids (également pondéré)
        initial_weights = np.ones(num_assets) / num_assets
    
        # Optimisation
        result = minimize(objective, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints)
    
        # Retour des poids optimaux
        return pd.Series(result.x, index=train_X.columns)
    
    # Optimisation des poids pour le proxy avec la combinaison corrélation + Tracking Error
    proxy_weights_combined = optimize_proxy_weights_combined(train_X_reduced, train_y, lambda_corr=0.7)  # Ajuster lambda_corr
    
    # Calcul des rendements cumulés pour le proxy combiné
    proxy_train_cumulative_combined = (1 + train_X_reduced.dot(proxy_weights_combined)).cumprod()
    proxy_test_cumulative_combined = (1 + test_X_reduced.dot(proxy_weights_combined)).cumprod()
    
    # Graphiques pour le proxy combiné
    plt.figure(figsize=(12, 6))
    plt.plot(proxy_train_cumulative_combined.index, proxy_train_cumulative_combined, label="Proxy Lambda_corr (Training)", color="green")
    plt.plot(train_btc_cumulative.index, train_btc_cumulative, label="Bitcoin (Training)", color="orange")
    plt.title("Proxy Lambda_corr vs Bitcoin (Training)")
    plt.xlabel("Date")
    plt.ylabel("Valeur Cumulée")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(proxy_test_cumulative_combined.index, proxy_test_cumulative_combined, label="Proxy Lambda_corr (Testing)", color="purple")
    plt.plot(test_btc_cumulative.index, test_btc_cumulative, label="Bitcoin (Testing)", color="orange")
    plt.title("Proxy Lambda_corr vs Bitcoin (Testing)")
    plt.xlabel("Date")
    plt.ylabel("Valeur Cumulée")
    plt.legend()
    plt.tight_layout()
    plt.show()


# 7. CALCUL DES MÉTRIQUES
    def calculate_metrics(cumulative_returns, daily_returns, benchmark_returns=None):
        total_return = cumulative_returns[-1] - 1
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (daily_returns.mean() * 252) / volatility
        downside_returns = daily_returns[daily_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (daily_returns.mean() * 252) / downside_deviation
        rolling_max = cumulative_returns.cummax()
        drawdown = cumulative_returns / rolling_max - 1
        max_drawdown = drawdown.min()
        calmar_ratio = (cumulative_returns[-1] ** (252 / len(daily_returns)) - 1) / abs(max_drawdown)
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
    
    # Calcul des métriques pour chaque modèle
    models = {
        "Portefeuille Rebalancé (Training)": (train_rebalanced, train_rebalanced.pct_change().dropna(), train_y),
        "Portefeuille Rebalancé (Testing)": (test_rebalanced, test_rebalanced.pct_change().dropna(), test_y),
        "Régression Linéaire (Training)": (train_replicated_cumulative, train_replicated_cumulative.pct_change().dropna(), train_y),
        "Régression Linéaire (Testing)": (test_replicated_cumulative, test_replicated_cumulative.pct_change().dropna(), test_y),
        "Random Forest (Training)": (rf_train_cumulative, rf_train_cumulative.pct_change().dropna(), train_y),
        "Random Forest (Testing)": (rf_test_cumulative, rf_test_cumulative.pct_change().dropna(), test_y),
        "Gradient Boosting (Training)": (gb_train_cumulative, gb_train_cumulative.pct_change().dropna(), train_y),
        "Gradient Boosting (Testing)": (gb_test_cumulative, gb_test_cumulative.pct_change().dropna(), test_y),
        "Proxy Index classique (Training)": (proxy_train_cumulative, proxy_train_cumulative.pct_change().dropna(), train_y),
        "Proxy Index classique  (Testing)": (proxy_test_cumulative, proxy_test_cumulative.pct_change().dropna(), test_y),
        "Proxy Régression Linéaire (Training)": (proxy_train_cumulative, train_X.dot(proxy_weights_optimized).pct_change().dropna(), train_y),
        "Proxy Régression Linéaire (Testing)": (proxy_test_cumulative, test_X.dot(proxy_weights_optimized).pct_change().dropna(), test_y),
        "Proxy Minimisation Tracking Error (Training)": (proxy_train_cumulative, train_X.dot(proxy_weights_optimized).pct_change().dropna(), train_y),
        "Proxy Minimisation Tracking Error (Testing)": (proxy_test_cumulative, test_X.dot(proxy_weights_optimized).pct_change().dropna(), test_y),
        "Proxy Corrélés (Training)": (proxy_train_cumulative_reduced, train_X_reduced.dot(proxy_weights_reduced).pct_change().dropna(), train_y),
        "Proxy Corrélés (Testing)": (proxy_test_cumulative_reduced, test_X_reduced.dot(proxy_weights_reduced).pct_change().dropna(), test_y),
        "Proxy Lambda_Corr (Training)": (proxy_train_cumulative_combined, train_X_reduced.dot(proxy_weights_combined).pct_change().dropna(), train_y),
        "Proxy Lambda_Corr (Testing)": (proxy_test_cumulative_combined, test_X_reduced.dot(proxy_weights_combined).pct_change().dropna(), test_y),
        "Bitcoin (Training)": (train_btc_cumulative, train_y, None),
        "Bitcoin (Testing)": (test_btc_cumulative, test_y, None),
    }
    
    # Affichage des métriques pour chaque modèle
    print("=== MÉTRIQUES DE PERFORMANCE ===")
    for model_name, (cumulative, daily, benchmark) in models.items():
        metrics = calculate_metrics(cumulative, daily, benchmark)
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2%}" if value is not None else f"  {metric}: NA")


#8 - Plots Récapitulatifs 
    # Plot récapitulatif - Training Set
    plt.figure(figsize=(14, 8))
    
    # Ajouter Bitcoin
    plt.plot(train_btc_cumulative.index, train_btc_cumulative, label="Bitcoin (Training)", color="orange", linewidth=2)
    
    # Ajouter les modèles
    plt.plot(train_rebalanced.index, train_rebalanced, label="Portefeuille Rebalancé", color="blue")
    plt.plot(train_replicated_cumulative.index, train_replicated_cumulative, label="Régression Linéaire", color="green")
    plt.plot(rf_train_cumulative.index, rf_train_cumulative, label="Random Forest", color="purple")
    plt.plot(gb_train_cumulative.index, gb_train_cumulative, label="Gradient Boosting", color="cyan")
    plt.plot(proxy_train_cumulative.index, proxy_train_cumulative, label="Proxy Index Classique", color="red")
    plt.plot(proxy_train_cumulative_reduced.index, proxy_train_cumulative_reduced, label="Proxy Corrélés", color="magenta")
    plt.plot(proxy_train_cumulative_combined.index, proxy_train_cumulative_combined, label="Proxy Lambda_Corr", color="brown")
    
    # Configuration du graphique
    plt.title("Comparaison des Performances Cumulées - Training Set")
    plt.xlabel("Date")
    plt.ylabel("Valeur Cumulée")
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), title="Modèles")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot récapitulatif - Testing Set
    plt.figure(figsize=(14, 8))
    
    # Ajouter Bitcoin
    plt.plot(test_btc_cumulative.index, test_btc_cumulative, label="Bitcoin (Testing)", color="orange", linewidth=2)
    
    # Ajouter les modèles
    plt.plot(test_rebalanced.index, test_rebalanced, label="Portefeuille Rebalancé", color="blue")
    plt.plot(test_replicated_cumulative.index, test_replicated_cumulative, label="Régression Linéaire", color="green")
    plt.plot(rf_test_cumulative.index, rf_test_cumulative, label="Random Forest", color="purple")
    plt.plot(gb_test_cumulative.index, gb_test_cumulative, label="Gradient Boosting", color="cyan")
    plt.plot(proxy_test_cumulative.index, proxy_test_cumulative, label="Proxy Index Classique", color="red")
    plt.plot(proxy_test_cumulative_reduced.index, proxy_test_cumulative_reduced, label="Proxy Corrélés", color="magenta")
    plt.plot(proxy_test_cumulative_combined.index, proxy_test_cumulative_combined, label="Proxy Lambda_Corr", color="brown")
    
    # Configuration du graphique
    plt.title("Comparaison des Performances Cumulées - Testing Set")
    plt.xlabel("Date")
    plt.ylabel("Valeur Cumulée")
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), title="Modèles")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    
    # Préparer la matrice des rendements journaliers pour les corrélations
    correlation_matrix = pd.DataFrame({
        "Bitcoin": train_y,
        "Portefeuille Rebalancé": train_rebalanced.pct_change(),
        "Random Forest": rf_train_cumulative.pct_change(),
        "Gradient Boosting": gb_train_cumulative.pct_change(),
        "Proxy Index Classique": proxy_train_cumulative.pct_change(),
        "Proxy Régression Linéaire": train_replicated_cumulative.pct_change(),
        "Proxy Minimisation TE": train_X.dot(proxy_weights_optimized).pct_change(),
        "Proxy Corrélés": train_X_reduced.dot(proxy_weights_reduced).pct_change(),
        "Proxy Lambda_Corr": train_X_reduced.dot(proxy_weights_combined).pct_change(),
    }).corr()
    
    # Tracer la heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, cbar_kws={"label": "Corrélation"})
    plt.title("Corrélations des Rendements Journaliers (Training Set)")
    plt.tight_layout()
    plt.show()



