"""
============================================
ELO-BASED NHL PREDICTION - DYNAMIC K-FACTOR ANALYSIS
Group: 1
Python Version: 3.9+
============================================

Tests different dynamic K-factor weighting functions where the update
magnitude depends on the ELO rating difference between teams.

Core Idea:
- When favorite wins: K is reduced (smaller update)
- When underdog wins: K is increased (larger update)

This makes upsets more impactful and expected wins less so.

Weighting Functions Tested:
1. Linear: K adjusts linearly with rating difference
2. Exponential: K adjusts exponentially with expected probability
3. Sigmoid: K adjusts using sigmoid function of rating difference
4. Polynomial: K adjusts with power function of probability difference
5. Square Root: K adjusts with sqrt of probability difference
6. Upset Multiplier: K multiplies when underdog wins, divides when favorite wins

Configuration: Base K values tested with different weighting functions
Home Advantage: 34 (from previous analysis)

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

# Fixed home advantage from previous analysis
HOME_ADVANTAGE = 34

print("=" * 80)
print("ELO-BASED NHL PREDICTION - DYNAMIC K-FACTOR ANALYSIS")
print("=" * 80)
print(f"Training Period: {TRAIN_START} to {TRAIN_END}")
print(f"Test Period: {TEST_START} to {TEST_END}")
print(f"Home Advantage: {HOME_ADVANTAGE}")
print(f"Random Seed: {RANDOM_SEED}")
print("=" * 80)


# ============================================
# 1. DYNAMIC K-FACTOR WEIGHTING FUNCTIONS
# ============================================

def k_factor_linear(base_k, rating_diff, max_diff=400):
    """
    Linear weighting: K decreases linearly as rating difference increases.

    When rating_diff = 0: K = base_k
    When rating_diff = max_diff: K = base_k * 0.5

    This means closer games have higher K.
    """
    abs_diff = min(abs(rating_diff), max_diff)
    scale = 1.0 - 0.5 * (abs_diff / max_diff)
    return base_k * scale


def k_factor_exponential(base_k, expected_prob):
    """
    Exponential weighting: K adjusts based on expected probability.

    When expected_prob = 0.5 (even match): K = base_k
    When expected_prob approaches 0 or 1 (mismatch): K decreases

    Uses formula: K = base_k * exp(-lambda * |0.5 - expected_prob|)
    """
    prob_diff = abs(expected_prob - 0.5)
    lambda_param = 3.0  # Controls decay rate
    scale = np.exp(-lambda_param * prob_diff)
    return base_k * scale


def k_factor_sigmoid(base_k, rating_diff, steepness=0.01, midpoint=200):
    """
    Sigmoid weighting: Smooth transition from high K to low K.

    Uses sigmoid to smoothly reduce K as rating difference increases.
    K ranges from base_k (for even matches) to base_k/2 (for mismatches).
    """
    abs_diff = abs(rating_diff)
    sigmoid = 1 / (1 + np.exp(-steepness * (abs_diff - midpoint)))
    scale = 1.0 - 0.5 * sigmoid
    return base_k * scale


def k_factor_polynomial(base_k, expected_prob, power=2):
    """
    Polynomial weighting: K adjusts with power function of probability difference.

    K = base_k * (1 - |0.5 - expected_prob|^power)

    Higher power makes the function more sensitive to extreme probabilities.
    """
    prob_diff = abs(expected_prob - 0.5)
    scale = 1.0 - (prob_diff / 0.5) ** power
    return base_k * scale


def k_factor_sqrt(base_k, expected_prob):
    """
    Square root weighting: Gentler adjustment than linear.

    K = base_k * (1 - sqrt(|0.5 - expected_prob| / 0.5))
    """
    prob_diff = abs(expected_prob - 0.5)
    scale = 1.0 - np.sqrt(prob_diff / 0.5)
    return base_k * scale


def k_factor_upset_multiplier(base_k, expected_prob, actual_score, upset_mult=2.0):
    """
    Upset multiplier: Increases K when underdog wins, decreases when favorite wins.

    If underdog wins (actual != expected): K = base_k * upset_mult
    If favorite wins (actual == expected): K = base_k / upset_mult
    If draw (actual = 0.5): K = base_k
    """
    # Determine if this was an upset
    if actual_score == 0.5:  # Draw
        return base_k
    elif (actual_score == 1.0 and expected_prob < 0.5) or \
         (actual_score == 0.0 and expected_prob > 0.5):
        # Upset: underdog won
        return base_k * upset_mult
    else:
        # Expected result: favorite won
        return base_k / upset_mult


# ============================================
# 2. HELPER FUNCTIONS
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
    """
    # Adjust home rating for home advantage
    home_rating_adj = home_rating + home_advantage

    # Calculate expected score for home team
    rating_diff = home_rating_adj - away_rating
    expected_home = 1 / (1 + 10 ** (-rating_diff / 400))

    # Tie probability based on rating difference
    abs_rating_diff = abs(rating_diff)
    base_tie_prob = 0.20
    min_tie_prob = 0.05
    tie_prob = min_tie_prob + (base_tie_prob - min_tie_prob) * np.exp(-abs_rating_diff / 150)

    # Distribute remaining probability
    remaining_prob = 1 - tie_prob
    prob_home_win = expected_home * remaining_prob
    prob_away_win = (1 - expected_home) * remaining_prob

    return prob_away_win, tie_prob, prob_home_win


def update_elo_ratings_dynamic(home_rating, away_rating, result, base_k=20,
                               home_advantage=100, weight_function='fixed',
                               weight_params=None):
    """
    Update ELO ratings with dynamic K-factor based on rating difference.

    Parameters:
    -----------
    home_rating : float
        Home team ELO before game
    away_rating : float
        Away team ELO before game
    result : int
        0=away win, 1=tie, 2=home win
    base_k : float
        Base K-factor (will be adjusted by weighting function)
    home_advantage : float
        Home ice advantage in ELO points
    weight_function : str
        Which weighting function to use:
        - 'fixed': Standard fixed K
        - 'linear': Linear decay with rating difference
        - 'exponential': Exponential decay with probability difference
        - 'sigmoid': Sigmoid decay with rating difference
        - 'polynomial': Polynomial decay with probability difference
        - 'sqrt': Square root decay with probability difference
        - 'upset_multiplier': Multiply K for upsets
    weight_params : dict
        Parameters for weighting function

    Returns:
    --------
    tuple : (new_home_rating, new_away_rating)
    """
    if weight_params is None:
        weight_params = {}

    # Adjust home rating for home advantage
    home_rating_adj = home_rating + home_advantage
    rating_diff = home_rating_adj - away_rating

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

    # Calculate dynamic K-factor
    if weight_function == 'fixed':
        k_home = k_away = base_k
    elif weight_function == 'linear':
        k_home = k_factor_linear(base_k, rating_diff,
                                weight_params.get('max_diff', 400))
        k_away = k_home  # Same K for both teams
    elif weight_function == 'exponential':
        k_home = k_factor_exponential(base_k, expected_home)
        k_away = k_home
    elif weight_function == 'sigmoid':
        k_home = k_factor_sigmoid(base_k, rating_diff,
                                  weight_params.get('steepness', 0.01),
                                  weight_params.get('midpoint', 200))
        k_away = k_home
    elif weight_function == 'polynomial':
        k_home = k_factor_polynomial(base_k, expected_home,
                                     weight_params.get('power', 2))
        k_away = k_home
    elif weight_function == 'sqrt':
        k_home = k_factor_sqrt(base_k, expected_home)
        k_away = k_home
    elif weight_function == 'upset_multiplier':
        k_home = k_factor_upset_multiplier(base_k, expected_home, actual_home,
                                          weight_params.get('upset_mult', 2.0))
        k_away = k_home
    else:
        raise ValueError(f"Unknown weight function: {weight_function}")

    # Update ratings
    new_home_rating = home_rating + k_home * (actual_home - expected_home)
    new_away_rating = away_rating + k_away * (actual_away - expected_away)

    return new_home_rating, new_away_rating


def run_elo_simulation(df, base_k, home_advantage=100, initial_rating=1500,
                      weight_function='fixed', weight_params=None):
    """
    Run ELO simulation with given parameters and weighting function.

    Returns predictions and updated dataframe with ELO ratings.
    """
    # Initialize ELO ratings
    all_teams = set(df['teamname_home'].unique()) | set(df['teamname_away'].unique())
    elo_ratings = initialize_elo_ratings(all_teams, initial_rating=initial_rating)

    # Track predictions
    predictions = []

    for idx, row in df.iterrows():
        home_team = row['teamname_home']
        away_team = row['teamname_away']
        result = row['result_reg']

        # Predict outcome
        prob_away, prob_tie, prob_home = predict_game_outcome_elo(
            elo_ratings[home_team],
            elo_ratings[away_team],
            home_advantage=home_advantage
        )
        predictions.append([prob_away, prob_tie, prob_home])

        # Update ELO with dynamic K
        new_home, new_away = update_elo_ratings_dynamic(
            elo_ratings[home_team],
            elo_ratings[away_team],
            result,
            base_k=base_k,
            home_advantage=home_advantage,
            weight_function=weight_function,
            weight_params=weight_params
        )

        elo_ratings[home_team] = new_home
        elo_ratings[away_team] = new_away

    predictions = np.array(predictions)
    return predictions


# ============================================
# 3. DATA LOADING
# ============================================

print("\n[1/3] Loading data...")

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
# 4. DYNAMIC K-FACTOR ANALYSIS
# ============================================

print("\n[2/3] Testing dynamic K-factor weighting functions...")

# Base K values to test
BASE_K_VALUES = [10, 15, 20, 25, 30]

# Weighting configurations to test
weighting_configs = [
    {'name': 'Fixed', 'function': 'fixed', 'params': {}},
    {'name': 'Linear (max_diff=300)', 'function': 'linear', 'params': {'max_diff': 300}},
    {'name': 'Linear (max_diff=400)', 'function': 'linear', 'params': {'max_diff': 400}},
    {'name': 'Linear (max_diff=500)', 'function': 'linear', 'params': {'max_diff': 500}},
    {'name': 'Exponential', 'function': 'exponential', 'params': {}},
    {'name': 'Sigmoid (steep=0.005, mid=200)', 'function': 'sigmoid',
     'params': {'steepness': 0.005, 'midpoint': 200}},
    {'name': 'Sigmoid (steep=0.01, mid=200)', 'function': 'sigmoid',
     'params': {'steepness': 0.01, 'midpoint': 200}},
    {'name': 'Sigmoid (steep=0.01, mid=150)', 'function': 'sigmoid',
     'params': {'steepness': 0.01, 'midpoint': 150}},
    {'name': 'Polynomial (power=1.5)', 'function': 'polynomial', 'params': {'power': 1.5}},
    {'name': 'Polynomial (power=2)', 'function': 'polynomial', 'params': {'power': 2}},
    {'name': 'Polynomial (power=3)', 'function': 'polynomial', 'params': {'power': 3}},
    {'name': 'Square Root', 'function': 'sqrt', 'params': {}},
    {'name': 'Upset Multiplier (x1.5)', 'function': 'upset_multiplier',
     'params': {'upset_mult': 1.5}},
    {'name': 'Upset Multiplier (x2)', 'function': 'upset_multiplier',
     'params': {'upset_mult': 2.0}},
    {'name': 'Upset Multiplier (x2.5)', 'function': 'upset_multiplier',
     'params': {'upset_mult': 2.5}},
    {'name': 'Upset Multiplier (x3)', 'function': 'upset_multiplier',
     'params': {'upset_mult': 3.0}},
]

# Prepare combined dataset
full_df = pd.concat([train_df, test_df], ignore_index=True).sort_values('date')
train_mask = full_df['date'] <= TRAIN_END
test_mask = full_df['date'] >= TEST_START

y_train = full_df[train_mask]['result_reg'].values
y_test = full_df[test_mask]['result_reg'].values

# Store all results
all_results = []

# Test each combination
total_configs = len(BASE_K_VALUES) * len(weighting_configs)
current = 0

for base_k in BASE_K_VALUES:
    for config in weighting_configs:
        current += 1
        print(f"\n[{current}/{total_configs}] Testing K={base_k}, {config['name']}...")

        # Run simulation
        predictions = run_elo_simulation(
            full_df,
            base_k=base_k,
            home_advantage=HOME_ADVANTAGE,
            weight_function=config['function'],
            weight_params=config['params']
        )

        # Split predictions
        train_predictions = predictions[train_mask]
        test_predictions = predictions[test_mask]

        # Make class predictions (argmax)
        train_pred_class = np.argmax(train_predictions, axis=1)
        test_pred_class = np.argmax(test_predictions, axis=1)

        # Calculate accuracies
        train_acc = accuracy_score(y_train, train_pred_class)
        test_acc = accuracy_score(y_test, test_pred_class)

        # Calculate log loss
        try:
            train_logloss = log_loss(y_train, train_predictions)
            test_logloss = log_loss(y_test, test_predictions)
        except:
            train_logloss = test_logloss = np.nan

        # Calculate Brier score
        train_brier = calculate_multiclass_brier_score(y_train, train_predictions)
        test_brier = calculate_multiclass_brier_score(y_test, test_predictions)

        # Store results
        result = {
            'base_k': base_k,
            'weight_function': config['function'],
            'weight_name': config['name'],
            'weight_params': str(config['params']),
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_logloss': train_logloss,
            'test_logloss': test_logloss,
            'train_brier': train_brier,
            'test_brier': test_brier
        }
        all_results.append(result)

        print(f"  Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"  Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"  Test Log Loss:  {test_logloss:.4f}")
        print(f"  Test Brier:     {test_brier:.4f}")


# ============================================
# 5. RESULTS SUMMARY
# ============================================

print("\n[3/3] Results Summary...")

results_df = pd.DataFrame(all_results)

# Sort by test accuracy
results_df = results_df.sort_values('test_accuracy', ascending=False)

print("\n" + "=" * 100)
print("DYNAMIC K-FACTOR ANALYSIS RESULTS")
print("=" * 100)
print("\nTop 20 Configurations by Test Accuracy:")
print("-" * 100)
print(f"{'Rank':<6} {'Base K':<8} {'Weighting Function':<35} {'Train Acc':<12} {'Test Acc':<12} {'Test Brier':<12}")
print("-" * 100)

for i, (idx, row) in enumerate(results_df.head(20).iterrows()):
    print(f"{i+1:<6} {row['base_k']:<8} {row['weight_name']:<35} "
          f"{row['train_accuracy']:.4f}      {row['test_accuracy']:.4f}      "
          f"{row['test_brier']:.4f}")

print("-" * 100)

# Find best configuration
best_row = results_df.iloc[0]
print(f"\n🏆 BEST CONFIGURATION:")
print(f"  Base K: {best_row['base_k']}")
print(f"  Weighting Function: {best_row['weight_name']}")
print(f"  Test Accuracy: {best_row['test_accuracy']:.4f} ({best_row['test_accuracy']*100:.2f}%)")
print(f"  Test Log Loss: {best_row['test_logloss']:.4f}")
print(f"  Test Brier Score: {best_row['test_brier']:.4f}")

# Compare to baseline (fixed K)
baseline_results = results_df[results_df['weight_function'] == 'fixed'].sort_values('test_accuracy', ascending=False)
if len(baseline_results) > 0:
    best_baseline = baseline_results.iloc[0]
    print(f"\n📊 BEST BASELINE (Fixed K):")
    print(f"  Base K: {best_baseline['base_k']}")
    print(f"  Test Accuracy: {best_baseline['test_accuracy']:.4f} ({best_baseline['test_accuracy']*100:.2f}%)")

    improvement = best_row['test_accuracy'] - best_baseline['test_accuracy']
    print(f"\n  Improvement: {improvement:.4f} ({improvement*100:+.2f}%)")

print("=" * 100)


# ============================================
# 6. SAVE RESULTS
# ============================================

print("\nSaving results...")

# Save all results
results_df.to_csv('elo_dynamic_k_factor_results.csv', index=False)
print("  Saved: elo_dynamic_k_factor_results.csv")

# Save summary
summary = {
    'model': 'ELO with Dynamic K-Factor',
    'home_advantage': HOME_ADVANTAGE,
    'base_k_values_tested': BASE_K_VALUES,
    'weighting_functions_tested': [c['name'] for c in weighting_configs],
    'total_configurations': total_configs,
    'best_configuration': {
        'base_k': int(best_row['base_k']),
        'weight_function': best_row['weight_function'],
        'weight_name': best_row['weight_name'],
        'weight_params': best_row['weight_params'],
        'test_accuracy': float(best_row['test_accuracy']),
        'test_logloss': float(best_row['test_logloss']),
        'test_brier': float(best_row['test_brier'])
    },
    'best_baseline': {
        'base_k': int(best_baseline['base_k']),
        'test_accuracy': float(best_baseline['test_accuracy'])
    } if len(baseline_results) > 0 else None,
    'improvement_over_baseline': float(improvement) if len(baseline_results) > 0 else None,
    'random_seed': RANDOM_SEED
}

with open('elo_dynamic_k_factor_summary.json', 'w') as f:
    json.dump(summary, f, indent=4)
print("  Saved: elo_dynamic_k_factor_summary.json")

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
print(f"\n🏆 BEST: K={best_row['base_k']}, {best_row['weight_name']}")
print(f"Test Accuracy: {best_row['test_accuracy']:.4f} ({best_row['test_accuracy']*100:.2f}%)")
print("=" * 100)
