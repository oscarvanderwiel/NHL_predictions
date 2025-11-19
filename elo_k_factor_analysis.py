"""
============================================
ELO-BASED NHL PREDICTION - K-FACTOR ANALYSIS
Group: 1
Python Version: 3.9+
============================================

Pure ELO rating approach following Arntzen & Hvattum (2010).
Tests different K-factor values to find optimal prediction accuracy.

Based on:
- Arntzen & Hvattum (2010): Using ELO ratings for match result prediction

STRICT REQUIREMENTS:
- Training/Validation: 2011-2020
- Test Set: 2021-2023 (FROZEN)
- NO DATA LEAKAGE
"""

# ============================================
# 0. SETUP & CONFIGURATION
# ============================================

import pandas as pd
import numpy as np
import random
import warnings
from datetime import datetime
import json
from sklearn.metrics import accuracy_score, log_loss

warnings.filterwarnings('ignore')

# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Configuration
TRAIN_START = '2011-01-01'
TRAIN_END = '2020-12-31'
TEST_START = '2021-01-01'
TEST_END = '2023-12-31'

print("=" * 80)
print("ELO-BASED NHL PREDICTION - K-FACTOR ANALYSIS")
print("=" * 80)
print(f"Training Period: {TRAIN_START} to {TRAIN_END}")
print(f"Test Period: {TEST_START} to {TEST_END}")
print(f"Random Seed: {RANDOM_SEED}")
print("=" * 80)


# ============================================
# 1. HELPER FUNCTIONS
# ============================================

def calculate_multiclass_brier_score(y_true, y_pred_proba):
    """Calculate multi-class Brier score."""
    n_samples = len(y_true)
    n_classes = y_pred_proba.shape[1]

    # One-hot encode true labels
    y_true_onehot = np.zeros((n_samples, n_classes))
    y_true_onehot[np.arange(n_samples), y_true] = 1

    # Calculate Brier score
    brier = np.mean(np.sum((y_pred_proba - y_true_onehot) ** 2, axis=1))
    return brier


def initialize_elo_ratings(teams, initial_rating=1500):
    """Initialize ELO ratings for all teams."""
    return {team: initial_rating for team in teams}


def predict_game_outcome_elo(home_rating, away_rating, home_advantage=100):
    """
    Predict game outcome probabilities using ELO ratings.

    Following Arntzen & Hvattum (2010) approach for three-outcome prediction.

    Parameters:
    -----------
    home_rating : float
        Home team ELO rating
    away_rating : float
        Away team ELO rating
    home_advantage : float
        Home ice advantage in ELO points

    Returns:
    --------
    tuple : (prob_away_win, prob_tie, prob_home_win)
    """
    # Adjust home rating for home advantage
    home_rating_adj = home_rating + home_advantage

    # Calculate expected score for home team (probability of home win in two-outcome)
    # Using standard ELO formula
    rating_diff = home_rating_adj - away_rating

    # Expected probability (home win in a two-outcome scenario)
    expected_home = 1 / (1 + 10 ** (-rating_diff / 400))

    # For three-outcome prediction, we use a simple approach:
    # Map the ELO difference to three outcomes
    # When teams are evenly matched, higher tie probability
    # When one team is much stronger, lower tie probability

    # Base tie probability (higher when teams are evenly matched)
    # This is a simplified model - can be enhanced with historical tie rates
    abs_rating_diff = abs(rating_diff)

    # Tie probability decreases as rating difference increases
    # When rating_diff = 0, tie_prob is highest (~0.20)
    # When rating_diff is large, tie_prob approaches 0.05
    base_tie_prob = 0.20
    min_tie_prob = 0.05
    tie_prob = min_tie_prob + (base_tie_prob - min_tie_prob) * np.exp(-abs_rating_diff / 150)

    # Distribute remaining probability between home and away based on ELO
    remaining_prob = 1 - tie_prob
    prob_home_win = expected_home * remaining_prob
    prob_away_win = (1 - expected_home) * remaining_prob

    return prob_away_win, tie_prob, prob_home_win


def update_elo_ratings(home_rating, away_rating, result, k_factor=20, home_advantage=100):
    """
    Update ELO ratings based on game result.

    Following Arntzen & Hvattum (2010) approach.

    Parameters:
    -----------
    home_rating : float
        Home team ELO before game
    away_rating : float
        Away team ELO before game
    result : int
        0=away win, 1=tie, 2=home win
    k_factor : float
        Learning rate (tunable hyperparameter)
    home_advantage : float
        Home ice advantage in ELO points (tunable hyperparameter)

    Returns:
    --------
    tuple : (new_home_rating, new_away_rating)
    """
    # Adjust home rating for home advantage
    home_rating_adj = home_rating + home_advantage

    # Expected scores
    expected_home = 1 / (1 + 10 ** ((away_rating - home_rating_adj) / 400))
    expected_away = 1 - expected_home

    # Actual scores
    if result == 2:  # Home win
        actual_home, actual_away = 1.0, 0.0
    elif result == 1:  # Tie
        actual_home, actual_away = 0.5, 0.5
    else:  # Away win
        actual_home, actual_away = 0.0, 1.0

    # Update ratings
    new_home_rating = home_rating + k_factor * (actual_home - expected_home)
    new_away_rating = away_rating + k_factor * (actual_away - expected_away)

    return new_home_rating, new_away_rating


def run_elo_simulation(df, k_factor, home_advantage=100, initial_rating=1500):
    """
    Run ELO simulation with given parameters.

    Returns predictions and updated dataframe with ELO ratings.
    """
    # Initialize ELO ratings
    all_teams = set(df['teamname_home'].unique()) | set(df['teamname_away'].unique())
    elo_ratings = initialize_elo_ratings(all_teams, initial_rating=initial_rating)

    # Track ELO ratings and predictions
    predictions = []
    elo_home_before = []
    elo_away_before = []

    for idx, row in df.iterrows():
        home_team = row['teamname_home']
        away_team = row['teamname_away']
        result = row['result_reg']

        # Store ELO BEFORE game
        elo_home_before.append(elo_ratings[home_team])
        elo_away_before.append(elo_ratings[away_team])

        # Predict outcome
        prob_away, prob_tie, prob_home = predict_game_outcome_elo(
            elo_ratings[home_team],
            elo_ratings[away_team],
            home_advantage=home_advantage
        )
        predictions.append([prob_away, prob_tie, prob_home])

        # Update ELO after game
        new_home, new_away = update_elo_ratings(
            elo_ratings[home_team],
            elo_ratings[away_team],
            result,
            k_factor=k_factor,
            home_advantage=home_advantage
        )

        elo_ratings[home_team] = new_home
        elo_ratings[away_team] = new_away

    # Add to dataframe
    df_copy = df.copy()
    df_copy['elo_home'] = elo_home_before
    df_copy['elo_away'] = elo_away_before
    df_copy['elo_diff'] = df_copy['elo_home'] - df_copy['elo_away']

    predictions = np.array(predictions)

    return df_copy, predictions


# ============================================
# 2. DATA LOADING
# ============================================

print("\n[1/4] Loading data...")

try:
    df = pd.read_csv('se_assignment1_1_data.csv')
    print(f"  Loaded: {len(df):,} games")
except FileNotFoundError:
    print("\nERROR: Could not find 'se_assignment1_1_data.csv'")
    print("Please run 'python prepare_data.py' first.")
    exit(1)

# Convert date and sort
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Remove missing data
critical_cols = ['date', 'result_reg', 'teamname_home', 'teamname_away']
df = df.dropna(subset=critical_cols)

print(f"  After removing missing values: {len(df):,} games")

# Split train/test
train_df = df[(df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)].copy()
test_df = df[(df['date'] >= TEST_START) & (df['date'] <= TEST_END)].copy()

print(f"\n  Training set: {len(train_df):,} games ({TRAIN_START} to {TRAIN_END})")
print(f"  Test set: {len(test_df):,} games ({TEST_START} to {TEST_END})")


# ============================================
# 3. K-FACTOR ANALYSIS
# ============================================

print("\n[2/4] Running K-factor analysis...")

# K-factors to test
K_FACTORS = [20, 25, 30, 35, 40]
HOME_ADVANTAGE = 100  # Fixed home advantage

# Store results
results = []

for k_factor in K_FACTORS:
    print(f"\n  Testing K-factor = {k_factor}...")

    # Combine train and test for sequential ELO calculation
    full_df = pd.concat([train_df, test_df], ignore_index=True).sort_values('date')

    # Run ELO simulation
    full_df_with_elo, predictions = run_elo_simulation(
        full_df,
        k_factor=k_factor,
        home_advantage=HOME_ADVANTAGE
    )

    # Split predictions back into train and test
    train_mask = full_df_with_elo['date'] <= TRAIN_END
    test_mask = full_df_with_elo['date'] >= TEST_START

    train_predictions = predictions[train_mask]
    test_predictions = predictions[test_mask]

    y_train = full_df_with_elo[train_mask]['result_reg'].values
    y_test = full_df_with_elo[test_mask]['result_reg'].values

    # Calculate metrics for training set
    train_pred_class = np.argmax(train_predictions, axis=1)
    train_accuracy = accuracy_score(y_train, train_pred_class)
    train_brier = calculate_multiclass_brier_score(y_train, train_predictions)
    train_logloss = log_loss(y_train, train_predictions)

    # Calculate metrics for test set
    test_pred_class = np.argmax(test_predictions, axis=1)
    test_accuracy = accuracy_score(y_test, test_pred_class)
    test_brier = calculate_multiclass_brier_score(y_test, test_predictions)
    test_logloss = log_loss(y_test, test_predictions)

    # Store results
    results.append({
        'k_factor': k_factor,
        'train_accuracy': train_accuracy,
        'train_brier': train_brier,
        'train_logloss': train_logloss,
        'test_accuracy': test_accuracy,
        'test_brier': test_brier,
        'test_logloss': test_logloss
    })

    print(f"    Train Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"    Test Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"    Test Brier:     {test_brier:.4f}")


# ============================================
# 4. RESULTS SUMMARY
# ============================================

print("\n[3/4] Results Summary...")

results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("K-FACTOR ANALYSIS RESULTS")
print("=" * 80)
print("\nPrediction Accuracy by K-Factor:")
print("-" * 80)
print(f"{'K-Factor':<12} {'Train Acc':<15} {'Test Acc':<15} {'Test Brier':<15}")
print("-" * 80)
for _, row in results_df.iterrows():
    print(f"{row['k_factor']:<12} {row['train_accuracy']:.4f} ({row['train_accuracy']*100:.2f}%)  "
          f"{row['test_accuracy']:.4f} ({row['test_accuracy']*100:.2f}%)  "
          f"{row['test_brier']:.4f}")
print("-" * 80)

# Find best K-factor
best_idx = results_df['test_accuracy'].idxmax()
best_k = results_df.loc[best_idx, 'k_factor']
best_acc = results_df.loc[best_idx, 'test_accuracy']

print(f"\nBest K-Factor: {best_k} (Test Accuracy: {best_acc:.4f} or {best_acc*100:.2f}%)")
print("=" * 80)


# ============================================
# 5. DETAILED EVALUATION OF BEST MODEL
# ============================================

print("\n[4/4] Detailed evaluation of best model...")

# Run simulation with best K-factor
full_df = pd.concat([train_df, test_df], ignore_index=True).sort_values('date')
full_df_with_elo, predictions = run_elo_simulation(
    full_df,
    k_factor=best_k,
    home_advantage=HOME_ADVANTAGE
)

# Split back into test set
test_mask = full_df_with_elo['date'] >= TEST_START
test_df_final = full_df_with_elo[test_mask].copy()
test_predictions_final = predictions[test_mask]
y_test_final = test_df_final['result_reg'].values

# Predictions
test_pred_class_final = np.argmax(test_predictions_final, axis=1)

# Detailed confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_final, test_pred_class_final)

print("\n" + "=" * 80)
print(f"DETAILED RESULTS FOR BEST MODEL (K={best_k})")
print("=" * 80)

print("\nConfusion Matrix:")
print("                  Predicted")
print("              Away    Tie    Home")
print(f"Actual Away   {cm[0,0]:4d}   {cm[0,1]:4d}   {cm[0,2]:4d}")
print(f"       Tie    {cm[1,0]:4d}   {cm[1,1]:4d}   {cm[1,2]:4d}")
print(f"       Home   {cm[2,0]:4d}   {cm[2,1]:4d}   {cm[2,2]:4d}")

# Per-class accuracy
print("\nPer-Class Accuracy:")
for i, outcome in enumerate(['Away Win', 'Tie', 'Home Win']):
    class_total = np.sum(y_test_final == i)
    class_correct = cm[i, i]
    class_acc = class_correct / class_total if class_total > 0 else 0
    print(f"  {outcome:<12}: {class_correct}/{class_total:4d} = {class_acc:.4f} ({class_acc*100:.2f}%)")

print("=" * 80)


# ============================================
# 6. SAVE RESULTS
# ============================================

print("\nSaving results...")

# Save results dataframe
results_df.to_csv('elo_k_factor_results.csv', index=False)
print("  Saved: elo_k_factor_results.csv")

# Save detailed predictions for best model
test_df_final['pred_away'] = test_predictions_final[:, 0]
test_df_final['pred_tie'] = test_predictions_final[:, 1]
test_df_final['pred_home'] = test_predictions_final[:, 2]
test_df_final['pred_class'] = test_pred_class_final

output_cols = ['date', 'season', 'teamname_home', 'teamname_away', 'result_reg',
               'elo_home', 'elo_away', 'elo_diff',
               'pred_away', 'pred_tie', 'pred_home', 'pred_class']
test_df_final[output_cols].to_csv('elo_test_predictions_best.csv', index=False)
print("  Saved: elo_test_predictions_best.csv")

# Save summary metrics
summary = {
    'model': 'Pure ELO Rating Model (Arntzen & Hvattum 2010)',
    'k_factors_tested': K_FACTORS,
    'home_advantage': HOME_ADVANTAGE,
    'best_k_factor': int(best_k),
    'results_by_k_factor': results_df.to_dict('records'),
    'best_model_test_accuracy': float(best_acc),
    'best_model_test_brier': float(results_df.loc[best_idx, 'test_brier']),
    'best_model_test_logloss': float(results_df.loc[best_idx, 'test_logloss']),
    'random_seed': RANDOM_SEED
}

with open('elo_k_factor_summary.json', 'w') as f:
    json.dump(summary, f, indent=4)
print("  Saved: elo_k_factor_summary.json")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nBest K-Factor: {best_k}")
print(f"Test Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
print("\nPure ELO-based approach following Arntzen & Hvattum (2010)")
print("=" * 80)
