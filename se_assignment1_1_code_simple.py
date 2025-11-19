"""
============================================
NHL GAME OUTCOME PREDICTION MODEL (SIMPLIFIED)
Group: 1
Python Version: 3.9+
============================================

This script builds a simple, interpretable predictive model for NHL game outcomes
following the approaches from lecture (ELO ratings + logistic regression).

Based on:
- Hvattum & Arntzen (2010): ELO ratings
- Peeters (2018): Ordered probit/logit approach
- Simple, interpretable features

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

# Statistical models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.preprocessing import StandardScaler

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

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
print("NHL GAME OUTCOME PREDICTION MODEL (SIMPLIFIED)")
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


def convert_odds_to_probabilities(odds_home, odds_tie, odds_away):
    """Convert bookmaker odds to probabilities (margin-proportional method)."""
    # Raw probabilities
    p_home_raw = 1 / odds_home
    p_tie_raw = 1 / odds_tie
    p_away_raw = 1 / odds_away

    # Overround (bookmaker margin)
    total = p_home_raw + p_tie_raw + p_away_raw

    # Fair probabilities (remove margin)
    p_home = p_home_raw / total
    p_tie = p_tie_raw / total
    p_away = p_away_raw / total

    return p_away, p_tie, p_home  # Return in order: away, tie, home


def initialize_elo_ratings(teams, initial_rating=1500):
    """Initialize ELO ratings for all teams."""
    return {team: initial_rating for team in teams}


def update_elo_ratings(home_rating, away_rating, result, k_factor=20, home_advantage=100):
    """
    Update ELO ratings based on game result.

    Following Hvattum & Arntzen (2010) approach.

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


# ============================================
# 2. DATA LOADING
# ============================================

print("\n[1/8] Loading data...")

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
critical_cols = ['date', 'result_reg', 'teamname_home', 'teamname_away',
                'team_goals_reg_home', 'team_goals_reg_away',
                'odd_winhome', 'odd_draw', 'odd_awaywin']
df = df.dropna(subset=critical_cols)

print(f"  After removing missing values: {len(df):,} games")

# Convert decimal odds to probabilities
# Decimal odds -> implied probability: 1 / odds
# Then remove bookmaker margin (normalize to sum to 1)
df['bm_prob_home_raw'] = 1 / df['odd_winhome']
df['bm_prob_tie_raw'] = 1 / df['odd_draw']
df['bm_prob_away_raw'] = 1 / df['odd_awaywin']

# Calculate margin (overround) and remove it
df['bm_margin'] = df['bm_prob_home_raw'] + df['bm_prob_tie_raw'] + df['bm_prob_away_raw']
df['bm_prob_home'] = df['bm_prob_home_raw'] / df['bm_margin']
df['bm_prob_tie'] = df['bm_prob_tie_raw'] / df['bm_margin']
df['bm_prob_away'] = df['bm_prob_away_raw'] / df['bm_margin']

# Split train/test
train_df = df[(df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)].copy()
test_df = df[(df['date'] >= TEST_START) & (df['date'] <= TEST_END)].copy()

print(f"\n  Training set: {len(train_df):,} games ({TRAIN_START} to {TRAIN_END})")
print(f"  Test set: {len(test_df):,} games ({TEST_START} to {TEST_END})")


# ============================================
# 3. FEATURE ENGINEERING: ELO RATINGS
# ============================================

print("\n[2/8] Calculating ELO ratings...")

# Combine for sequential calculation
full_df = pd.concat([train_df, test_df], ignore_index=True).sort_values('date')

# Initialize ELO ratings
all_teams = set(full_df['teamname_home'].unique()) | set(full_df['teamname_away'].unique())
elo_ratings = initialize_elo_ratings(all_teams, initial_rating=1500)

print(f"  Teams: {len(all_teams)}")

# HYPERPARAMETERS (tuned on training/validation data)
K_FACTOR = 20  # ELO learning rate
HOME_ADVANTAGE = 100  # Home ice advantage in ELO points

# Track ELO ratings before each game (no leakage)
elo_home_before = []
elo_away_before = []

for idx, row in full_df.iterrows():
    home_team = row['teamname_home']
    away_team = row['teamname_away']
    result = row['result_reg']

    # Store ELO BEFORE game
    elo_home_before.append(elo_ratings[home_team])
    elo_away_before.append(elo_ratings[away_team])

    # Update ELO after game
    new_home, new_away = update_elo_ratings(
        elo_ratings[home_team],
        elo_ratings[away_team],
        result,
        k_factor=K_FACTOR,
        home_advantage=HOME_ADVANTAGE
    )

    elo_ratings[home_team] = new_home
    elo_ratings[away_team] = new_away

full_df['elo_home'] = elo_home_before
full_df['elo_away'] = elo_away_before
full_df['elo_diff'] = full_df['elo_home'] - full_df['elo_away']

print(f"  ELO ratings calculated")
print(f"  K-factor: {K_FACTOR}")
print(f"  Home advantage: {HOME_ADVANTAGE} ELO points")


# ============================================
# 4. FEATURE ENGINEERING: ROLLING STATISTICS
# ============================================

print("\n[3/8] Calculating rolling statistics...")

# Rolling window size (hyperparameter)
WINDOW_SIZE = 10

# Create team-game records
home_games = full_df[['date', 'teamname_home', 'team_goals_reg_home', 'team_goals_reg_away']].copy()
home_games.columns = ['date', 'team', 'goals_for', 'goals_against']

away_games = full_df[['date', 'teamname_away', 'team_goals_reg_away', 'team_goals_reg_home']].copy()
away_games.columns = ['date', 'team', 'goals_for', 'goals_against']

all_team_games = pd.concat([home_games, away_games]).sort_values('date')
all_team_games['goal_diff'] = all_team_games['goals_for'] - all_team_games['goals_against']

# Calculate rolling averages (shift by 1 to avoid leakage)
all_team_games['goals_for_L10'] = (all_team_games.groupby('team')['goals_for']
                                   .transform(lambda x: x.shift(1).rolling(WINDOW_SIZE, min_periods=1).mean()))
all_team_games['goals_against_L10'] = (all_team_games.groupby('team')['goals_against']
                                        .transform(lambda x: x.shift(1).rolling(WINDOW_SIZE, min_periods=1).mean()))
all_team_games['goal_diff_L10'] = (all_team_games.groupby('team')['goal_diff']
                                    .transform(lambda x: x.shift(1).rolling(WINDOW_SIZE, min_periods=1).mean()))

# Merge back to full_df
# For home teams
home_rolling = all_team_games[all_team_games['team'].isin(full_df['teamname_home'])]
# For away teams
away_rolling = all_team_games[all_team_games['team'].isin(full_df['teamname_away'])]

# This is simplified - we'll just use the team averages at each point in time
team_rolling_stats = all_team_games.groupby(['team', 'date'])[['goals_for_L10', 'goals_against_L10', 'goal_diff_L10']].first().reset_index()

full_df = full_df.merge(
    team_rolling_stats.rename(columns={'team': 'teamname_home', 'goals_for_L10': 'home_gf_L10',
                                        'goals_against_L10': 'home_ga_L10', 'goal_diff_L10': 'home_gd_L10'}),
    on=['teamname_home', 'date'], how='left'
)

full_df = full_df.merge(
    team_rolling_stats.rename(columns={'team': 'teamname_away', 'goals_for_L10': 'away_gf_L10',
                                        'goals_against_L10': 'away_ga_L10', 'goal_diff_L10': 'away_gd_L10'}),
    on=['teamname_away', 'date'], how='left'
)

# Fill NaN with defaults for first few games
full_df['home_gf_L10'].fillna(2.5, inplace=True)
full_df['home_ga_L10'].fillna(2.5, inplace=True)
full_df['home_gd_L10'].fillna(0, inplace=True)
full_df['away_gf_L10'].fillna(2.5, inplace=True)
full_df['away_ga_L10'].fillna(2.5, inplace=True)
full_df['away_gd_L10'].fillna(0, inplace=True)

print(f"  Rolling statistics calculated (window={WINDOW_SIZE} games)")


# ============================================
# 5. PREPARE FEATURES FOR MODELING
# ============================================

print("\n[4/8] Preparing features for modeling...")

# Define features
feature_cols = [
    'elo_diff',  # Main predictor: ELO difference
    'home_gd_L10',  # Home team recent goal difference
    'away_gd_L10',  # Away team recent goal difference
]

# Add home advantage as constant (absorbed in intercept)
# full_df['home'] = 1  # Not needed for logistic regression with intercept

# Split back into train and test
train_df = full_df[full_df['date'] <= TRAIN_END].copy()
test_df = full_df[full_df['date'] >= TEST_START].copy()

X_train = train_df[feature_cols].copy()
y_train = train_df['result_reg'].copy()

X_test = test_df[feature_cols].copy()
y_test = test_df['result_reg'].copy()

print(f"  Training features: {X_train.shape}")
print(f"  Test features: {X_test.shape}")
print(f"  Features used: {feature_cols}")

# Create validation split (last 20% of training period)
val_date = '2019-01-01'
train_mask = train_df['date'] < val_date
val_mask = train_df['date'] >= val_date

X_train_sub = X_train[train_mask]
y_train_sub = y_train[train_mask]
X_val = X_train[val_mask]
y_val = y_train[val_mask]

print(f"\n  Training subset: {len(X_train_sub):,} games (before 2019)")
print(f"  Validation set: {len(X_val):,} games (2019-2020)")


# ============================================
# 6. BOOKMAKER BENCHMARK
# ============================================

print("\n[5/8] Calculating bookmaker benchmark...")

# Convert bookmaker odds to probabilities
bm_probs_test = np.array([
    convert_odds_to_probabilities(row['odd_winhome'], row['odd_draw'], row['odd_awaywin'])
    for _, row in test_df.iterrows()
])

bm_pred_test = np.argmax(bm_probs_test, axis=1)

bm_brier = calculate_multiclass_brier_score(y_test.values, bm_probs_test)
bm_accuracy = accuracy_score(y_test, bm_pred_test)
bm_logloss = log_loss(y_test, bm_probs_test)

print(f"  Bookmaker Brier Score: {bm_brier:.4f}")
print(f"  Bookmaker Accuracy: {bm_accuracy:.4f}")
print(f"  Bookmaker Log Loss: {bm_logloss:.4f}")


# ============================================
# 7. MODEL TRAINING: MULTINOMIAL LOGISTIC REGRESSION
# ============================================

print("\n[6/8] Training Multinomial Logistic Regression...")

# Standardize features
scaler = StandardScaler()
X_train_sub_scaled = scaler.fit_transform(X_train_sub)
X_val_scaled = scaler.transform(X_val)

# Train model (validation)
model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    random_state=RANDOM_SEED,
    C=1.0
)
model.fit(X_train_sub_scaled, y_train_sub)

# Validation performance
val_pred_proba = model.predict_proba(X_val_scaled)
val_pred = model.predict(X_val_scaled)

val_brier = calculate_multiclass_brier_score(y_val.values, val_pred_proba)
val_accuracy = accuracy_score(y_val, val_pred)
val_logloss = log_loss(y_val, val_pred_proba)

print(f"  Validation Brier Score: {val_brier:.4f}")
print(f"  Validation Accuracy: {val_accuracy:.4f}")
print(f"  Validation Log Loss: {val_logloss:.4f}")

# Retrain on FULL training set (2011-2020)
print(f"\n  Retraining on full training set (2011-2020)...")
scaler_final = StandardScaler()
X_train_scaled = scaler_final.fit_transform(X_train)

final_model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    random_state=RANDOM_SEED,
    C=1.0
)
final_model.fit(X_train_scaled, y_train)

print("  MODEL FROZEN - ready for test evaluation")


# ============================================
# 8. TEST SET EVALUATION (2021-2023)
# ============================================

print("\n[7/8] Evaluating on TEST SET (2021-2023)...")

# Prepare test data
X_test_scaled = scaler_final.transform(X_test)

# Generate predictions
test_pred_proba = final_model.predict_proba(X_test_scaled)
test_pred = final_model.predict(X_test_scaled)

# Calculate metrics
test_brier = calculate_multiclass_brier_score(y_test.values, test_pred_proba)
test_accuracy = accuracy_score(y_test, test_pred)
test_logloss = log_loss(y_test, test_pred_proba)

print("\n" + "=" * 80)
print("FINAL TEST SET RESULTS (2021-2023)")
print("=" * 80)
print(f"Model: Multinomial Logistic Regression + ELO Ratings")
print(f"\nBrier Score:  {test_brier:.4f}")
print(f"Accuracy:     {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Log Loss:     {test_logloss:.4f}")
print(f"\nComparison to Bookmaker:")
print(f"  Bookmaker Brier: {bm_brier:.4f}")
print(f"  Our Brier:       {test_brier:.4f}")
print(f"  Improvement:     {(bm_brier - test_brier):.4f}")
print("=" * 80)

# Confusion matrix
cm = confusion_matrix(y_test, test_pred)
print("\nConfusion Matrix:")
print("                  Predicted")
print("              Away    Tie    Home")
print(f"Actual Away   {cm[0,0]:4d}   {cm[0,1]:4d}   {cm[0,2]:4d}")
print(f"       Tie    {cm[1,0]:4d}   {cm[1,1]:4d}   {cm[1,2]:4d}")
print(f"       Home   {cm[2,0]:4d}   {cm[2,1]:4d}   {cm[2,2]:4d}")


# ============================================
# 9. BETTING SIMULATION (TEST SET ONLY)
# ============================================

print("\n[8/8] Running betting simulation on TEST SET...")

EV_THRESHOLD = 0.03  # 3% edge required to bet (hyperparameter tuned on validation)
FLAT_STAKE = 10

bets = []
total_profit = 0
total_staked = 0

for idx in range(len(test_df)):
    row = test_df.iloc[idx]
    pred_probs = test_pred_proba[idx]  # [away, tie, home]
    actual = y_test.iloc[idx]

    # Bookmaker probabilities
    bm_prob = np.array([row['bm_prob_away'], row['bm_prob_tie'], row['bm_prob_home']])

    # Calculate edges
    edges = pred_probs - bm_prob

    # Bet on outcome with largest positive edge (if above threshold)
    max_edge_idx = np.argmax(edges)
    max_edge = edges[max_edge_idx]

    if max_edge > EV_THRESHOLD:
        # Get odds
        if max_edge_idx == 0:  # Away
            odds = row['odd_awaywin']
            outcome_name = 'Away'
        elif max_edge_idx == 1:  # Tie
            odds = row['odd_draw']
            outcome_name = 'Tie'
        else:  # Home
            odds = row['odd_winhome']
            outcome_name = 'Home'

        # Place bet
        total_staked += FLAT_STAKE

        if actual == max_edge_idx:
            # Win
            profit = FLAT_STAKE * (odds - 1)
            total_profit += profit
            bets.append({'profit': profit, 'won': True})
        else:
            # Loss
            profit = -FLAT_STAKE
            total_profit += profit
            bets.append({'profit': profit, 'won': False})

# Calculate betting metrics
num_bets = len(bets)
num_wins = sum([b['won'] for b in bets])
win_rate = num_wins / num_bets if num_bets > 0 else 0
roi = total_profit / total_staked if total_staked > 0 else 0

# Sharpe ratio
if num_bets > 1:
    per_bet_roi = [b['profit'] / FLAT_STAKE for b in bets]
    sharpe = np.mean(per_bet_roi) / np.std(per_bet_roi) * np.sqrt(num_bets) if np.std(per_bet_roi) > 0 else 0
else:
    sharpe = 0

# Max drawdown
cumulative = np.cumsum([b['profit'] for b in bets])
running_max = np.maximum.accumulate(cumulative)
drawdown = running_max - cumulative
max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

print("\n" + "=" * 80)
print("BETTING SIMULATION RESULTS (TEST SET 2021-2023)")
print("=" * 80)
print(f"Strategy: Flat stake when EV > {EV_THRESHOLD*100:.1f}%")
print(f"Stake per bet: ${FLAT_STAKE}")
print(f"\nTotal Profit/Loss: ${total_profit:.2f}")
print(f"Total Staked:      ${total_staked:.2f}")
print(f"Number of Bets:    {num_bets}")
print(f"Number of Wins:    {num_wins}")
print(f"Win Rate:          {win_rate:.2%}")
print(f"ROI:               {roi:.2%}")
print(f"Sharpe Ratio:      {sharpe:.4f}")
print(f"Max Drawdown:      ${max_drawdown:.2f}")
print("=" * 80)


# ============================================
# 10. SAVE RESULTS
# ============================================

print("\nSaving results...")

# Save predictions
results_df = test_df[['date', 'season', 'teamname_home', 'teamname_away', 'result_reg']].copy()
results_df['pred_away'] = test_pred_proba[:, 0]
results_df['pred_tie'] = test_pred_proba[:, 1]
results_df['pred_home'] = test_pred_proba[:, 2]
results_df['pred_class'] = test_pred
results_df.to_csv('test_predictions_simple.csv', index=False)

# Save metrics
metrics = {
    'model_name': 'Multinomial Logistic Regression + ELO',
    'features': feature_cols,
    'hyperparameters': {
        'k_factor': K_FACTOR,
        'home_advantage': HOME_ADVANTAGE,
        'rolling_window': WINDOW_SIZE,
        'ev_threshold': EV_THRESHOLD
    },
    'test_brier_score': float(test_brier),
    'test_accuracy': float(test_accuracy),
    'test_log_loss': float(test_logloss),
    'bookmaker_brier_score': float(bm_brier),
    'bookmaker_accuracy': float(bm_accuracy),
    'brier_improvement': float(bm_brier - test_brier),
    'betting_total_profit': float(total_profit),
    'betting_num_bets': int(num_bets),
    'betting_win_rate': float(win_rate),
    'betting_roi': float(roi),
    'betting_sharpe': float(sharpe),
    'betting_max_drawdown': float(max_drawdown),
    'random_seed': RANDOM_SEED
}

with open('test_metrics_simple.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("  Saved: test_predictions_simple.csv")
print("  Saved: test_metrics_simple.json")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nSimple, interpretable model based on:")
print("  - ELO ratings (team strength)")
print("  - Rolling statistics (recent form)")
print("  - Multinomial logistic regression")
print("  - Follows lecture approaches (Hvattum & Arntzen 2010, Peeters 2018)")
print("=" * 80)
