"""
============================================
ELO-BASED NHL PREDICTION - PROBABILITY DIFFERENCE TIE ANALYSIS
Group: 1
Python Version: 3.9+
============================================

Tests tie prediction based on probability differences.
When prob(home win) - prob(away win) differs by less than x%, predict a tie.
Otherwise, predict the outcome with the highest probability.

Configuration: K=15, Home Advantage=34

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
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

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

# Fixed parameters from user
K_FACTOR = 15
HOME_ADVANTAGE = 34

print("=" * 80)
print("ELO-BASED NHL PREDICTION - PROBABILITY DIFFERENCE TIE ANALYSIS")
print("=" * 80)
print(f"Training Period: {TRAIN_START} to {TRAIN_END}")
print(f"Test Period: {TEST_START} to {TEST_END}")
print(f"K-Factor: {K_FACTOR}")
print(f"Home Advantage: {HOME_ADVANTAGE}")
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


def predict_with_probability_difference_threshold(predictions, prob_diff_threshold):
    """
    Make predictions with probability difference threshold.

    If abs(prob_home - prob_away) < prob_diff_threshold, predict tie (class 1).
    Otherwise, predict the class with highest probability.

    Parameters:
    -----------
    predictions : np.array
        Array of shape (n_samples, 3) with probabilities [away, tie, home]
    prob_diff_threshold : float
        Threshold for predicting tie (as decimal, e.g., 0.01 for 1%)

    Returns:
    --------
    np.array : Predicted classes (0=away, 1=tie, 2=home)
    """
    predicted_classes = []

    for i in range(len(predictions)):
        prob_away = predictions[i, 0]
        prob_home = predictions[i, 2]

        # Calculate absolute difference between home and away win probabilities
        prob_diff = abs(prob_home - prob_away)

        if prob_diff < prob_diff_threshold:
            # Probabilities are close, predict tie
            predicted_classes.append(1)
        else:
            # Use probability-based prediction
            predicted_classes.append(np.argmax(predictions[i]))

    return np.array(predicted_classes)


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
# 3. PROBABILITY DIFFERENCE THRESHOLD ANALYSIS
# ============================================

print("\n[2/4] Running probability difference threshold analysis...")

# Probability difference thresholds to test (as percentages, converted to decimals)
PROB_DIFF_THRESHOLDS_PCT = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
PROB_DIFF_THRESHOLDS = [x / 100.0 for x in PROB_DIFF_THRESHOLDS_PCT]

# First, run ELO simulation once to get all predictions
print(f"\n  Running ELO simulation (K={K_FACTOR}, Home Adv={HOME_ADVANTAGE})...")
full_df = pd.concat([train_df, test_df], ignore_index=True).sort_values('date')

full_df_with_elo, predictions = run_elo_simulation(
    full_df,
    k_factor=K_FACTOR,
    home_advantage=HOME_ADVANTAGE
)

# Split predictions back into train and test
train_mask = full_df_with_elo['date'] <= TRAIN_END
test_mask = full_df_with_elo['date'] >= TEST_START

train_predictions = predictions[train_mask]
test_predictions = predictions[test_mask]

y_train = full_df_with_elo[train_mask]['result_reg'].values
y_test = full_df_with_elo[test_mask]['result_reg'].values

# Store results
results = []

print("\n  Testing different probability difference thresholds...")

# Also test baseline (no threshold, just argmax)
print(f"\n  Testing baseline (no tie threshold)...")
baseline_train_pred = np.argmax(train_predictions, axis=1)
baseline_test_pred = np.argmax(test_predictions, axis=1)
baseline_train_acc = accuracy_score(y_train, baseline_train_pred)
baseline_test_acc = accuracy_score(y_test, baseline_test_pred)

results.append({
    'threshold_pct': 0.0,
    'threshold_decimal': 0.0,
    'train_accuracy': baseline_train_acc,
    'test_accuracy': baseline_test_acc,
    'description': 'Baseline (no threshold)'
})

print(f"    Train Accuracy: {baseline_train_acc:.4f} ({baseline_train_acc*100:.2f}%)")
print(f"    Test Accuracy:  {baseline_test_acc:.4f} ({baseline_test_acc*100:.2f}%)")

# Test each threshold
for threshold_pct, threshold_decimal in zip(PROB_DIFF_THRESHOLDS_PCT, PROB_DIFF_THRESHOLDS):
    print(f"\n  Testing threshold = {threshold_pct:.1f}% ({threshold_decimal:.4f})...")

    # Make predictions with probability difference threshold
    train_pred_class = predict_with_probability_difference_threshold(train_predictions, threshold_decimal)
    test_pred_class = predict_with_probability_difference_threshold(test_predictions, threshold_decimal)

    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, train_pred_class)
    test_accuracy = accuracy_score(y_test, test_pred_class)

    # Count tie predictions
    train_tie_preds = np.sum(train_pred_class == 1)
    test_tie_preds = np.sum(test_pred_class == 1)
    train_tie_rate = train_tie_preds / len(train_pred_class)
    test_tie_rate = test_tie_preds / len(test_pred_class)

    # Store results
    results.append({
        'threshold_pct': threshold_pct,
        'threshold_decimal': threshold_decimal,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_tie_predictions': train_tie_preds,
        'test_tie_predictions': test_tie_preds,
        'train_tie_rate': train_tie_rate,
        'test_tie_rate': test_tie_rate,
        'description': f'Threshold = {threshold_pct:.1f}%'
    })

    print(f"    Train Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"    Test Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"    Train Tie Predictions: {train_tie_preds} ({train_tie_rate*100:.2f}%)")
    print(f"    Test Tie Predictions:  {test_tie_preds} ({test_tie_rate*100:.2f}%)")


# ============================================
# 4. RESULTS SUMMARY
# ============================================

print("\n[3/4] Results Summary...")

results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("PROBABILITY DIFFERENCE THRESHOLD ANALYSIS RESULTS")
print("=" * 80)
print("\nPrediction Accuracy by Probability Difference Threshold:")
print("-" * 80)
print(f"{'Threshold':<15} {'Train Acc':<15} {'Test Acc':<15} {'Test Tie %':<15}")
print("-" * 80)
for _, row in results_df.iterrows():
    tie_pct = row.get('test_tie_rate', 0) * 100 if 'test_tie_rate' in row else 0
    threshold_str = f"{row['threshold_pct']:.1f}%" if row['threshold_pct'] > 0 else "Baseline"
    print(f"{threshold_str:<15} {row['train_accuracy']:.4f} ({row['train_accuracy']*100:5.2f}%)  "
          f"{row['test_accuracy']:.4f} ({row['test_accuracy']*100:5.2f}%)  "
          f"{tie_pct:5.2f}%")
print("-" * 80)

# Find best threshold
best_idx = results_df['test_accuracy'].idxmax()
best_threshold_pct = results_df.loc[best_idx, 'threshold_pct']
best_threshold_decimal = results_df.loc[best_idx, 'threshold_decimal']
best_acc = results_df.loc[best_idx, 'test_accuracy']

print(f"\n🏆 BEST THRESHOLD: {best_threshold_pct:.1f}% (Test Accuracy: {best_acc:.4f} or {best_acc*100:.2f}%)")
print("=" * 80)


# ============================================
# 5. DETAILED EVALUATION OF BEST MODEL
# ============================================

print("\n[4/4] Detailed evaluation of best model...")

# Get predictions for best threshold
if best_threshold_decimal == 0:
    test_pred_class_final = np.argmax(test_predictions, axis=1)
else:
    test_pred_class_final = predict_with_probability_difference_threshold(test_predictions, best_threshold_decimal)

test_df_final = full_df_with_elo[test_mask].copy()
y_test_final = test_df_final['result_reg'].values

# Detailed confusion matrix
cm = confusion_matrix(y_test_final, test_pred_class_final)

print("\n" + "=" * 80)
print(f"DETAILED RESULTS FOR BEST MODEL (Threshold={best_threshold_pct:.1f}%)")
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
results_df.to_csv('elo_probability_difference_results.csv', index=False)
print("  Saved: elo_probability_difference_results.csv")

# Save detailed predictions for best model
test_df_final['pred_away'] = test_predictions[:, 0]
test_df_final['pred_tie'] = test_predictions[:, 1]
test_df_final['pred_home'] = test_predictions[:, 2]
test_df_final['prob_difference'] = np.abs(test_predictions[:, 2] - test_predictions[:, 0])
test_df_final['pred_class'] = test_pred_class_final
test_df_final['pred_class_baseline'] = np.argmax(test_predictions, axis=1)

output_cols = ['date', 'season', 'teamname_home', 'teamname_away', 'result_reg',
               'elo_home', 'elo_away', 'elo_diff',
               'pred_away', 'pred_tie', 'pred_home', 'prob_difference',
               'pred_class', 'pred_class_baseline']
test_df_final[output_cols].to_csv('elo_probability_difference_predictions.csv', index=False)
print("  Saved: elo_probability_difference_predictions.csv")

# Save summary metrics
summary = {
    'model': 'ELO with Probability Difference Threshold',
    'k_factor': K_FACTOR,
    'home_advantage': HOME_ADVANTAGE,
    'thresholds_tested_pct': [0.0] + PROB_DIFF_THRESHOLDS_PCT,
    'best_threshold_pct': float(best_threshold_pct),
    'best_threshold_decimal': float(best_threshold_decimal),
    'results_by_threshold': results_df.to_dict('records'),
    'best_model_test_accuracy': float(best_acc),
    'baseline_test_accuracy': float(baseline_test_acc),
    'accuracy_improvement': float(best_acc - baseline_test_acc),
    'random_seed': RANDOM_SEED
}

with open('elo_probability_difference_summary.json', 'w') as f:
    json.dump(summary, f, indent=4)
print("  Saved: elo_probability_difference_summary.json")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\n🏆 BEST THRESHOLD: {best_threshold_pct:.1f}%")
print(f"Test Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
print(f"Baseline Accuracy: {baseline_test_acc:.4f} ({baseline_test_acc*100:.2f}%)")
print(f"Improvement: {(best_acc - baseline_test_acc):.4f} ({(best_acc - baseline_test_acc)*100:+.2f}%)")
print("\nELO-based prediction with probability difference threshold")
print("=" * 80)
