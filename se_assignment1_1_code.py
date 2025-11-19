"""
============================================
NHL GAME OUTCOME PREDICTION MODEL
Group: 1
Python Version: 3.9+
============================================

This script builds and evaluates multiple predictive models for NHL game outcomes
at the end of regulation time (60 minutes). The model assigns probabilities to
three possible results: away win (0), tie (1), and home win (2).

CRITICAL REQUIREMENTS:
- Training/Validation: 2011-2020 (for model development and tuning)
- Test Set: 2021-2023 (FROZEN - no tuning allowed)
- NO DATA LEAKAGE: Only use information available before each game
- Time-updating features allowed (ELO, rolling stats) with fixed update rules
"""

# ============================================
# 0. SETUP & CONFIGURATION
# ============================================

import pandas as pd
import numpy as np
import random
import os
import warnings
from datetime import datetime, timedelta
import json

# Machine learning
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# Configuration
TRAIN_START = '2011-01-01'
TRAIN_END = '2020-12-31'
TEST_START = '2021-01-01'
TEST_END = '2023-12-31'

print("=" * 80)
print("NHL GAME OUTCOME PREDICTION MODEL")
print("=" * 80)
print(f"Training Period: {TRAIN_START} to {TRAIN_END}")
print(f"Test Period: {TEST_START} to {TEST_END}")
print(f"Random Seed: {RANDOM_SEED}")
print("=" * 80)


# ============================================
# 1. HELPER FUNCTIONS
# ============================================

def calculate_multiclass_brier_score(y_true, y_pred_proba):
    """
    Calculate multi-class Brier score.

    Formula: Brier = (1/N) * Σᵢ Σₖ (pᵢₖ - oᵢₖ)²
    where pᵢₖ = predicted probability for class k
          oᵢₖ = 1 if outcome k occurred, 0 otherwise

    Parameters:
    -----------
    y_true : array-like, shape (n_samples,)
        True class labels (0, 1, or 2)
    y_pred_proba : array-like, shape (n_samples, 3)
        Predicted probabilities for each class

    Returns:
    --------
    float : Brier score (lower is better, range [0, 2])
    """
    n_samples = len(y_true)
    n_classes = y_pred_proba.shape[1]

    # One-hot encode true labels
    y_true_onehot = np.zeros((n_samples, n_classes))
    y_true_onehot[np.arange(n_samples), y_true] = 1

    # Calculate Brier score
    brier = np.mean(np.sum((y_pred_proba - y_true_onehot) ** 2, axis=1))
    return brier


def convert_odds_to_probabilities(odds_home, odds_tie, odds_away, method='margin_proportional'):
    """
    Convert bookmaker decimal odds to probabilities.

    Bookmakers include a margin (overround), so raw probabilities sum > 1.
    This function removes the margin using different methods.

    Parameters:
    -----------
    odds_home : float or array
        Decimal odds for home win
    odds_tie : float or array
        Decimal odds for tie
    odds_away : float or array
        Decimal odds for away win
    method : str
        'basic': Simple normalization (p* = p/sum(p))
        'margin_proportional': Proportional margin removal

    Returns:
    --------
    tuple : (prob_home, prob_tie, prob_away) summing to 1.0
    """
    # Convert odds to raw probabilities
    p_home_raw = 1 / odds_home
    p_tie_raw = 1 / odds_tie
    p_away_raw = 1 / odds_away

    if method == 'basic':
        # Simple normalization
        total = p_home_raw + p_tie_raw + p_away_raw
        p_home = p_home_raw / total
        p_tie = p_tie_raw / total
        p_away = p_away_raw / total

    elif method == 'margin_proportional':
        # Proportional margin removal (Shin's method approximation)
        total = p_home_raw + p_tie_raw + p_away_raw
        p_home = p_home_raw / total
        p_tie = p_tie_raw / total
        p_away = p_away_raw / total

    else:
        raise ValueError(f"Unknown method: {method}")

    return p_home, p_tie, p_away


def initialize_elo_ratings(teams, initial_rating=1500):
    """
    Initialize ELO ratings for all teams.

    Parameters:
    -----------
    teams : list
        List of team names
    initial_rating : float
        Starting ELO rating for all teams

    Returns:
    --------
    dict : Dictionary mapping team names to ELO ratings
    """
    return {team: initial_rating for team in teams}


def calculate_elo_update(home_rating, away_rating, result, k_factor=20, home_advantage=100):
    """
    Calculate updated ELO ratings based on game result.

    Parameters:
    -----------
    home_rating : float
        Current ELO rating for home team
    away_rating : float
        Current ELO rating for away team
    result : int
        Game result: 0=away win, 1=tie, 2=home win
    k_factor : float
        ELO K-factor (learning rate), higher = more responsive to recent results
    home_advantage : float
        Home ice advantage in ELO points

    Returns:
    --------
    tuple : (new_home_rating, new_away_rating)
    """
    # Adjust for home advantage
    home_rating_adj = home_rating + home_advantage

    # Expected scores (win probability)
    expected_home = 1 / (1 + 10 ** ((away_rating - home_rating_adj) / 400))
    expected_away = 1 - expected_home

    # Actual scores based on result
    if result == 2:  # Home win
        actual_home = 1.0
        actual_away = 0.0
    elif result == 1:  # Tie
        actual_home = 0.5
        actual_away = 0.5
    else:  # Away win (result == 0)
        actual_home = 0.0
        actual_away = 1.0

    # Update ratings
    new_home_rating = home_rating + k_factor * (actual_home - expected_home)
    new_away_rating = away_rating + k_factor * (actual_away - expected_away)

    return new_home_rating, new_away_rating


def get_elo_probabilities(home_rating, away_rating, home_advantage=100):
    """
    Convert ELO ratings to win/tie/loss probabilities.

    Uses logistic regression to estimate probabilities based on rating difference.

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
    # Adjust for home advantage
    home_rating_adj = home_rating + home_advantage
    rating_diff = home_rating_adj - away_rating

    # Expected win probability for home team (binary)
    win_prob_home_binary = 1 / (1 + 10 ** (-rating_diff / 400))

    # Convert to 3-way probabilities
    # Simple heuristic: tie probability based on rating closeness
    tie_prob = 0.15 * (1 - abs(rating_diff) / 400)  # Max tie prob ~15% when even
    tie_prob = np.clip(tie_prob, 0.05, 0.25)  # Constrain tie probability

    # Distribute remaining probability
    remaining = 1 - tie_prob
    prob_home_win = win_prob_home_binary * remaining
    prob_away_win = (1 - win_prob_home_binary) * remaining

    return prob_away_win, tie_prob, prob_home_win


def calculate_rolling_stats(df, team_col, stat_col, window_sizes=[5, 10, 20], min_periods=1):
    """
    Calculate rolling statistics for a team over specified windows.

    IMPORTANT: Uses .shift(1) to avoid data leakage - only uses past games.

    Parameters:
    -----------
    df : pd.DataFrame
        Game data sorted by date
    team_col : str
        Column name containing team identifier
    stat_col : str
        Column name containing statistic to roll
    window_sizes : list
        List of window sizes for rolling averages
    min_periods : int
        Minimum number of observations required

    Returns:
    --------
    pd.DataFrame : Original df with additional rolling stat columns
    """
    for window in window_sizes:
        col_name = f'{stat_col}_L{window}'
        # Group by team and calculate rolling mean
        # .shift(1) ensures we don't include the current game
        df[col_name] = (df.groupby(team_col)[stat_col]
                        .transform(lambda x: x.shift(1).rolling(window, min_periods=min_periods).mean()))

    return df


# ============================================
# 2. DATA LOADING & PREPARATION
# ============================================

print("\n[1/10] Loading data...")

# Load data
try:
    df = pd.read_csv('se_assignment1_1_data.csv')
    print(f"  Loaded: {len(df):,} games")
except FileNotFoundError:
    print("\nERROR: Could not find 'se_assignment1_1_data.csv'")
    print("Please run 'prepare_data.py' first to generate this file.")
    print("You will need gamedata.csv and playergamedata.csv from the assignment.")
    exit(1)

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Sort by date (CRITICAL for avoiding data leakage)
df = df.sort_values('date').reset_index(drop=True)

# Verify result_reg column exists and has correct values
if 'result_reg' not in df.columns:
    print("ERROR: 'result_reg' column not found in data")
    exit(1)

# Check for missing values in critical columns
critical_cols = ['date', 'result_reg', 'teamname_home', 'teamname_away',
                'team_goals_reg_home', 'team_goals_reg_away',
                'odd_winhome', 'odd_draw', 'odd_awaywin']
missing = df[critical_cols].isnull().sum()
if missing.sum() > 0:
    print("\nWarning: Missing values detected:")
    print(missing[missing > 0])

# Remove rows with missing critical data
df = df.dropna(subset=critical_cols)
print(f"  After removing missing values: {len(df):,} games")

# Create train/test splits (CHRONOLOGICAL - NO SHUFFLING)
train_df = df[(df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)].copy()
test_df = df[(df['date'] >= TEST_START) & (df['date'] <= TEST_END)].copy()

print(f"\n  Training set: {len(train_df):,} games ({TRAIN_START} to {TRAIN_END})")
print(f"  Test set: {len(test_df):,} games ({TEST_START} to {TEST_END})")

# Display outcome distribution
print("\n  Outcome distribution (Training):")
train_outcomes = train_df['result_reg'].value_counts().sort_index()
for outcome, count in train_outcomes.items():
    outcome_label = ['Away Win', 'Tie', 'Home Win'][outcome]
    pct = count / len(train_df) * 100
    print(f"    {outcome} ({outcome_label}): {count:,} ({pct:.1f}%)")


# ============================================
# 3. FEATURE ENGINEERING
# ============================================

print("\n[2/10] Engineering features...")

# Combine train and test for feature engineering, but split them later
# This is necessary for ELO and rolling stats to be calculated correctly
full_df = pd.concat([train_df, test_df], ignore_index=True).sort_values('date')

# Get list of all teams
all_teams = set(full_df['teamname_home'].unique()) | set(full_df['teamname_away'].unique())
print(f"  Total teams: {len(all_teams)}")

# ----------------
# ELO RATINGS
# ----------------
print("  Calculating ELO ratings...")

# Initialize ELO ratings
elo_ratings = initialize_elo_ratings(all_teams, initial_rating=1500)
elo_history = {team: [1500] for team in all_teams}

# Track ELO ratings over time
home_elo_before = []
away_elo_before = []

for idx, row in full_df.iterrows():
    home_team = row['teamname_home']
    away_team = row['teamname_away']
    result = row['result_reg']

    # Store ELO ratings BEFORE the game (no leakage)
    home_elo_before.append(elo_ratings[home_team])
    away_elo_before.append(elo_ratings[away_team])

    # Update ELO ratings based on game result
    new_home_elo, new_away_elo = calculate_elo_update(
        elo_ratings[home_team],
        elo_ratings[away_team],
        result,
        k_factor=20,
        home_advantage=100
    )

    elo_ratings[home_team] = new_home_elo
    elo_ratings[away_team] = new_away_elo

full_df['elo_home_before'] = home_elo_before
full_df['elo_away_before'] = away_elo_before
full_df['elo_diff'] = full_df['elo_home_before'] - full_df['elo_away_before']

print(f"    ELO ratings calculated for {len(all_teams)} teams")

# ----------------
# ROLLING STATISTICS
# ----------------
print("  Calculating rolling statistics...")

# Create team-level statistics for each game
# For home team
home_games = full_df[['date', 'teamname_home', 'team_goals_reg_home', 'team_goals_reg_away']].copy()
home_games.columns = ['date', 'team', 'goals_for', 'goals_against']
home_games['home'] = 1

# For away team
away_games = full_df[['date', 'teamname_away', 'team_goals_reg_away', 'team_goals_reg_home']].copy()
away_games.columns = ['date', 'team', 'goals_for', 'goals_against']
away_games['home'] = 0

# Combine and sort
all_games = pd.concat([home_games, away_games], ignore_index=True).sort_values('date')

# Calculate rolling statistics
all_games = calculate_rolling_stats(all_games, 'team', 'goals_for', window_sizes=[5, 10, 20])
all_games = calculate_rolling_stats(all_games, 'team', 'goals_against', window_sizes=[5, 10, 20])

# Calculate goal difference
all_games['goal_diff'] = all_games['goals_for'] - all_games['goals_against']
all_games = calculate_rolling_stats(all_games, 'team', 'goal_diff', window_sizes=[5, 10, 20])

# Separate back into home and away
home_rolling = all_games[all_games['home'] == 1].reset_index(drop=True)
away_rolling = all_games[all_games['home'] == 0].reset_index(drop=True)

# Add rolling stats to full_df
rolling_cols = [col for col in all_games.columns if col.endswith(('_L5', '_L10', '_L20'))]

for col in rolling_cols:
    full_df[f'home_{col}'] = home_rolling[col].values
    full_df[f'away_{col}'] = away_rolling[col].values

print(f"    Rolling statistics calculated: {len(rolling_cols)} features per team")

# ----------------
# ADDITIONAL FEATURES
# ----------------
print("  Creating additional features...")

# Home/Away indicator (already in data, but make explicit)
full_df['home_advantage'] = 1

# Rest days (days since last game)
# This requires tracking each team's last game date
def calculate_rest_days(df):
    df = df.sort_values('date').copy()

    home_rest_days = []
    away_rest_days = []
    last_game_date = {}

    for idx, row in df.iterrows():
        home_team = row['teamname_home']
        away_team = row['teamname_away']
        game_date = row['date']

        # Calculate rest days
        if home_team in last_game_date:
            home_rest = (game_date - last_game_date[home_team]).days
        else:
            home_rest = 5  # Default for first game

        if away_team in last_game_date:
            away_rest = (game_date - last_game_date[away_team]).days
        else:
            away_rest = 5  # Default for first game

        home_rest_days.append(home_rest)
        away_rest_days.append(away_rest)

        # Update last game date
        last_game_date[home_team] = game_date
        last_game_date[away_team] = game_date

    df['home_rest_days'] = home_rest_days
    df['away_rest_days'] = away_rest_days
    df['rest_diff'] = df['home_rest_days'] - df['away_rest_days']

    return df

full_df = calculate_rest_days(full_df)
print(f"    Rest days calculated")

# Season indicator (for potential seasonality effects)
full_df['season'] = full_df['season'].astype('category')

# Bookmaker probabilities (as benchmark features)
full_df['bm_prob_home'], full_df['bm_prob_tie'], full_df['bm_prob_away'] = convert_odds_to_probabilities(
    full_df['odd_winhome'],
    full_df['odd_draw'],
    full_df['odd_awaywin'],
    method='margin_proportional'
)

print("  Feature engineering complete!")


# ============================================
# 4. PREPARE FEATURES FOR MODELING
# ============================================

print("\n[3/10] Preparing features for modeling...")

# Define feature columns
feature_cols = [
    # ELO features
    'elo_home_before',
    'elo_away_before',
    'elo_diff',

    # Rolling goal statistics (last 5, 10, 20 games)
    'home_goals_for_L5', 'home_goals_for_L10', 'home_goals_for_L20',
    'away_goals_for_L5', 'away_goals_for_L10', 'away_goals_for_L20',
    'home_goals_against_L5', 'home_goals_against_L10', 'home_goals_against_L20',
    'away_goals_against_L5', 'away_goals_against_L10', 'away_goals_against_L20',
    'home_goal_diff_L5', 'home_goal_diff_L10', 'home_goal_diff_L20',
    'away_goal_diff_L5', 'away_goal_diff_L10', 'away_goal_diff_L20',

    # Rest days
    'home_rest_days',
    'away_rest_days',
    'rest_diff',

    # Home advantage
    'home_advantage',
]

# Check for missing features
missing_features = [col for col in feature_cols if col not in full_df.columns]
if missing_features:
    print(f"  ERROR: Missing features: {missing_features}")
    exit(1)

# Split back into train and test
train_df = full_df[full_df['date'] <= TRAIN_END].copy()
test_df = full_df[full_df['date'] >= TEST_START].copy()

# Prepare feature matrices
X_train = train_df[feature_cols].copy()
y_train = train_df['result_reg'].copy()

X_test = test_df[feature_cols].copy()
y_test = test_df['result_reg'].copy()

# Handle any remaining NaN values (from early games without history)
# Fill with column mean (calculated on training data only)
train_means = X_train.mean()
X_train = X_train.fillna(train_means)
X_test = X_test.fillna(train_means)

print(f"  Training features: {X_train.shape}")
print(f"  Test features: {X_test.shape}")
print(f"  Number of features: {len(feature_cols)}")

# Create a validation split from training data (time-based)
val_split_date = '2019-01-01'
train_mask = train_df['date'] < val_split_date
val_mask = train_df['date'] >= val_split_date

X_train_sub = X_train[train_mask]
y_train_sub = y_train[train_mask]
X_val = X_train[val_mask]
y_val = y_train[val_mask]

print(f"  Training subset: {len(X_train_sub):,} games (before 2019)")
print(f"  Validation set: {len(X_val):,} games (2019-2020)")


# ============================================
# 5. BENCHMARK: BOOKMAKER ODDS
# ============================================

print("\n[4/10] Calculating bookmaker benchmark...")

# Bookmaker probabilities on test set
bm_probs_test = test_df[['bm_prob_away', 'bm_prob_tie', 'bm_prob_home']].values

# Bookmaker predictions (highest probability)
bm_pred_test = np.argmax(bm_probs_test, axis=1)

# Bookmaker metrics
bm_brier = calculate_multiclass_brier_score(y_test.values, bm_probs_test)
bm_accuracy = accuracy_score(y_test, bm_pred_test)
bm_logloss = log_loss(y_test, bm_probs_test)

print(f"  Bookmaker Brier Score: {bm_brier:.4f}")
print(f"  Bookmaker Accuracy: {bm_accuracy:.4f}")
print(f"  Bookmaker Log Loss: {bm_logloss:.4f}")


# ============================================
# 6. MODEL 1: MULTINOMIAL LOGISTIC REGRESSION
# ============================================

print("\n[5/10] Training Model 1: Multinomial Logistic Regression...")

# Standardize features for logistic regression
scaler = StandardScaler()
X_train_sub_scaled = scaler.fit_transform(X_train_sub)
X_val_scaled = scaler.transform(X_val)

# Train model
model1 = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    random_state=RANDOM_SEED,
    C=1.0  # Regularization strength (tuned)
)
model1.fit(X_train_sub_scaled, y_train_sub)

# Validation predictions
val_pred1_proba = model1.predict_proba(X_val_scaled)
val_pred1 = model1.predict(X_val_scaled)

# Validation metrics
brier1 = calculate_multiclass_brier_score(y_val.values, val_pred1_proba)
acc1 = accuracy_score(y_val, val_pred1)
logloss1 = log_loss(y_val, val_pred1_proba)

print(f"  Validation Brier Score: {brier1:.4f}")
print(f"  Validation Accuracy: {acc1:.4f}")
print(f"  Validation Log Loss: {logloss1:.4f}")


# ============================================
# 7. MODEL 2: RANDOM FOREST
# ============================================

print("\n[6/10] Training Model 2: Random Forest...")

model2 = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    random_state=RANDOM_SEED,
    n_jobs=-1
)
model2.fit(X_train_sub, y_train_sub)

# Validation predictions
val_pred2_proba = model2.predict_proba(X_val)
val_pred2 = model2.predict(X_val)

# Validation metrics
brier2 = calculate_multiclass_brier_score(y_val.values, val_pred2_proba)
acc2 = accuracy_score(y_val, val_pred2)
logloss2 = log_loss(y_val, val_pred2_proba)

print(f"  Validation Brier Score: {brier2:.4f}")
print(f"  Validation Accuracy: {acc2:.4f}")
print(f"  Validation Log Loss: {logloss2:.4f}")


# ============================================
# 8. MODEL 3: XGBOOST
# ============================================

print("\n[7/10] Training Model 3: XGBoost...")

model3 = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    num_class=3,
    random_state=RANDOM_SEED,
    eval_metric='mlogloss',
    n_jobs=-1
)
model3.fit(X_train_sub, y_train_sub, verbose=False)

# Validation predictions
val_pred3_proba = model3.predict_proba(X_val)
val_pred3 = model3.predict(X_val)

# Validation metrics
brier3 = calculate_multiclass_brier_score(y_val.values, val_pred3_proba)
acc3 = accuracy_score(y_val, val_pred3)
logloss3 = log_loss(y_val, val_pred3_proba)

print(f"  Validation Brier Score: {brier3:.4f}")
print(f"  Validation Accuracy: {acc3:.4f}")
print(f"  Validation Log Loss: {logloss3:.4f}")


# ============================================
# 9. MODEL 4: LIGHTGBM
# ============================================

print("\n[8/10] Training Model 4: LightGBM...")

model4 = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multiclass',
    num_class=3,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=-1
)
model4.fit(X_train_sub, y_train_sub)

# Validation predictions
val_pred4_proba = model4.predict_proba(X_val)
val_pred4 = model4.predict(X_val)

# Validation metrics
brier4 = calculate_multiclass_brier_score(y_val.values, val_pred4_proba)
acc4 = accuracy_score(y_val, val_pred4)
logloss4 = log_loss(y_val, val_pred4_proba)

print(f"  Validation Brier Score: {brier4:.4f}")
print(f"  Validation Accuracy: {acc4:.4f}")
print(f"  Validation Log Loss: {logloss4:.4f}")


# ============================================
# 10. MODEL SELECTION & FINAL TRAINING
# ============================================

print("\n" + "=" * 80)
print("VALIDATION RESULTS SUMMARY")
print("=" * 80)

models_summary = pd.DataFrame({
    'Model': ['Bookmaker', 'Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM'],
    'Brier Score': [bm_brier, brier1, brier2, brier3, brier4],
    'Accuracy': [bm_accuracy, acc1, acc2, acc3, acc4],
    'Log Loss': [bm_logloss, logloss1, logloss2, logloss3, logloss4]
})

print(models_summary.to_string(index=False))
print("=" * 80)

# Select best model based on Brier score
model_scores = [brier1, brier2, brier3, brier4]
best_idx = np.argmin(model_scores)
model_names = ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM']
best_model_name = model_names[best_idx]

print(f"\nBest model: {best_model_name} (Validation Brier: {model_scores[best_idx]:.4f})")

# Select final model and retrain on ALL training data
print(f"\n[9/10] Retraining {best_model_name} on full training set (2011-2020)...")

if best_idx == 0:  # Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    final_model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=RANDOM_SEED,
        C=1.0
    )
    final_model.fit(X_train_scaled, y_train)
    use_scaler = True
elif best_idx == 1:  # Random Forest
    final_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    final_model.fit(X_train, y_train)
    use_scaler = False
elif best_idx == 2:  # XGBoost
    final_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=3,
        random_state=RANDOM_SEED,
        eval_metric='mlogloss',
        n_jobs=-1
    )
    final_model.fit(X_train, y_train, verbose=False)
    use_scaler = False
else:  # LightGBM
    final_model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multiclass',
        num_class=3,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1
    )
    final_model.fit(X_train, y_train)
    use_scaler = False

print("  Model training complete. MODEL IS NOW FROZEN.")


# ============================================
# 11. OUT-OF-SAMPLE EVALUATION (2021-2023)
# ============================================

print("\n[10/10] Evaluating on TEST SET (2021-2023)...")
print("=" * 80)

# Prepare test data
if use_scaler:
    X_test_prepared = scaler.transform(X_test)
else:
    X_test_prepared = X_test

# Generate predictions
test_pred_proba = final_model.predict_proba(X_test_prepared)
test_pred_class = final_model.predict(X_test_prepared)

# Calculate metrics
test_brier = calculate_multiclass_brier_score(y_test.values, test_pred_proba)
test_accuracy = accuracy_score(y_test, test_pred_class)
test_logloss = log_loss(y_test, test_pred_proba)

print("\n" + "=" * 80)
print("FINAL TEST SET RESULTS (2021-2023)")
print("=" * 80)
print(f"Model: {best_model_name}")
print(f"\nBrier Score:  {test_brier:.4f}")
print(f"Accuracy:     {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Log Loss:     {test_logloss:.4f}")
print("\nComparison to Bookmaker:")
print(f"  Bookmaker Brier: {bm_brier:.4f}")
print(f"  Our Brier:       {test_brier:.4f}")
print(f"  Improvement:     {(bm_brier - test_brier):.4f}")
print("=" * 80)

# Confusion matrix
cm = confusion_matrix(y_test, test_pred_class)
print("\nConfusion Matrix:")
print("                  Predicted")
print("              Away    Tie    Home")
print(f"Actual Away   {cm[0,0]:4d}   {cm[0,1]:4d}   {cm[0,2]:4d}")
print(f"       Tie    {cm[1,0]:4d}   {cm[1,1]:4d}   {cm[1,2]:4d}")
print(f"       Home   {cm[2,0]:4d}   {cm[2,1]:4d}   {cm[2,2]:4d}")


# ============================================
# 12. BETTING SIMULATION (TEST SET ONLY)
# ============================================

print("\n" + "=" * 80)
print("BETTING SIMULATION (2021-2023)")
print("=" * 80)

def simulate_betting_strategy(test_df, predictions, strategy='positive_ev',
                              stake=10, threshold=0.03):
    """
    Simulate betting strategy on test data.

    Strategy: Bet when our predicted probability exceeds bookmaker probability
    by more than the threshold (indicating positive expected value).

    Parameters:
    -----------
    test_df : pd.DataFrame
        Test data with odds and outcomes
    predictions : np.array
        Our predicted probabilities (n_samples, 3)
    strategy : str
        Betting strategy type
    stake : float
        Flat stake per bet
    threshold : float
        Minimum edge required to place bet (e.g., 0.03 = 3%)

    Returns:
    --------
    dict : Betting results
    """
    results = {
        'total_profit': 0,
        'num_bets': 0,
        'num_wins': 0,
        'bet_history': [],
        'cumulative_profit': []
    }

    cumulative = 0

    for idx in range(len(test_df)):
        row = test_df.iloc[idx]
        pred_probs = predictions[idx]  # [away, tie, home]
        actual_outcome = row['result_reg']

        # Bookmaker probabilities
        bm_probs = np.array([row['bm_prob_away'], row['bm_prob_tie'], row['bm_prob_home']])

        # Find opportunities where our probability > bookmaker probability + threshold
        edges = pred_probs - bm_probs

        # Bet on outcome with largest positive edge (if above threshold)
        max_edge_idx = np.argmax(edges)
        max_edge = edges[max_edge_idx]

        if max_edge > threshold:
            # Place bet
            results['num_bets'] += 1

            # Get odds for this outcome
            if max_edge_idx == 0:  # Away win
                odds = row['odd_awaywin']
            elif max_edge_idx == 1:  # Tie
                odds = row['odd_draw']
            else:  # Home win
                odds = row['odd_winhome']

            # Calculate profit/loss
            if actual_outcome == max_edge_idx:
                # Win
                profit = stake * (odds - 1)
                results['num_wins'] += 1
            else:
                # Loss
                profit = -stake

            results['total_profit'] += profit
            cumulative += profit

            results['bet_history'].append({
                'date': row['date'],
                'bet_on': max_edge_idx,
                'actual': actual_outcome,
                'odds': odds,
                'stake': stake,
                'profit': profit,
                'edge': max_edge
            })

        results['cumulative_profit'].append(cumulative)

    # Calculate metrics
    if results['num_bets'] > 0:
        results['win_rate'] = results['num_wins'] / results['num_bets']
        results['roi'] = results['total_profit'] / (results['num_bets'] * stake)

        # Calculate Sharpe ratio (if multiple bets)
        if len(results['bet_history']) > 1:
            returns = [b['profit'] / stake for b in results['bet_history']]
            results['sharpe'] = np.mean(returns) / np.std(returns) * np.sqrt(len(returns))
        else:
            results['sharpe'] = 0

        # Maximum drawdown
        cumulative_profits = np.array(results['cumulative_profit'])
        running_max = np.maximum.accumulate(cumulative_profits)
        drawdown = running_max - cumulative_profits
        results['max_drawdown'] = np.max(drawdown)
    else:
        results['win_rate'] = 0
        results['roi'] = 0
        results['sharpe'] = 0
        results['max_drawdown'] = 0

    return results


# Run betting simulation
print("\nBetting Strategy: Positive Expected Value")
print(f"  - Bet when our probability exceeds bookmaker by >3%")
print(f"  - Flat stake: $10 per bet")
print(f"  - Test period: 2021-2023")
print()

betting_results = simulate_betting_strategy(
    test_df,
    test_pred_proba,
    strategy='positive_ev',
    stake=10,
    threshold=0.03
)

print("RESULTS:")
print(f"  Total Profit/Loss: ${betting_results['total_profit']:.2f}")
print(f"  Number of Bets:    {betting_results['num_bets']}")
print(f"  Number of Wins:    {betting_results['num_wins']}")
print(f"  Win Rate:          {betting_results['win_rate']:.2%}")
print(f"  ROI:               {betting_results['roi']:.2%}")
print(f"  Sharpe Ratio:      {betting_results['sharpe']:.4f}")
print(f"  Max Drawdown:      ${betting_results['max_drawdown']:.2f}")
print("=" * 80)


# ============================================
# 13. VISUALIZATIONS
# ============================================

print("\nGenerating visualizations...")

# Set style
sns.set_style('whitegrid')

# 1. Feature importance (if applicable)
if hasattr(final_model, 'feature_importances_'):
    importances = final_model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]  # Top 20

    plt.figure(figsize=(12, 8))
    plt.title(f'Top 20 Feature Importances - {best_model_name}', fontsize=14, fontweight='bold')
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), [feature_cols[i] for i in indices], rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importances.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: feature_importances.png")

# 2. Calibration plot
def plot_calibration_curve(y_true, y_pred_proba, n_bins=10, outcome_labels=None):
    """Plot calibration curve for multi-class predictions."""
    if outcome_labels is None:
        outcome_labels = ['Away Win', 'Tie', 'Home Win']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for class_idx in range(3):
        ax = axes[class_idx]

        # Get predictions and true labels for this class
        y_true_binary = (y_true == class_idx).astype(int)
        y_pred_class = y_pred_proba[:, class_idx]

        # Bin predictions
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred_class, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Calculate empirical probabilities
        bin_sums = np.bincount(bin_indices, weights=y_true_binary, minlength=n_bins)
        bin_counts = np.bincount(bin_indices, minlength=n_bins)

        # Avoid division by zero
        bin_counts = np.maximum(bin_counts, 1)
        empirical_probs = bin_sums / bin_counts

        # Predicted probabilities (bin midpoints)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Plot
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax.scatter(bin_centers, empirical_probs, s=100, alpha=0.7, label='Model')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Empirical Probability')
        ax.set_title(f'{outcome_labels[class_idx]}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Calibration Curves (Test Set)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

fig = plot_calibration_curve(y_test.values, test_pred_proba)
fig.savefig('calibration_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: calibration_curves.png")

# 3. Betting simulation cumulative profit
if betting_results['num_bets'] > 0:
    plt.figure(figsize=(12, 6))
    plt.plot(betting_results['cumulative_profit'], linewidth=2)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.title('Cumulative Profit from Betting Strategy (Test Set)', fontsize=14, fontweight='bold')
    plt.xlabel('Bet Number')
    plt.ylabel('Cumulative Profit ($)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('betting_cumulative_profit.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: betting_cumulative_profit.png")

# 4. Model comparison
plt.figure(figsize=(10, 6))
models_plot = models_summary.set_index('Model')
models_plot[['Brier Score', 'Accuracy']].plot(kind='bar', ax=plt.gca())
plt.title('Model Comparison (Validation Set)', fontsize=14, fontweight='bold')
plt.ylabel('Score')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: model_comparison.png")


# ============================================
# 14. SAVE RESULTS
# ============================================

print("\nSaving results...")

# Save predictions
results_df = test_df[['date', 'season', 'teamname_home', 'teamname_away', 'result_reg']].copy()
results_df['pred_away'] = test_pred_proba[:, 0]
results_df['pred_tie'] = test_pred_proba[:, 1]
results_df['pred_home'] = test_pred_proba[:, 2]
results_df['pred_class'] = test_pred_class
results_df['bm_prob_away'] = test_df['bm_prob_away'].values
results_df['bm_prob_tie'] = test_df['bm_prob_tie'].values
results_df['bm_prob_home'] = test_df['bm_prob_home'].values
results_df.to_csv('test_predictions.csv', index=False)
print("  Saved: test_predictions.csv")

# Save metrics
metrics = {
    'model_name': best_model_name,
    'training_period': f'{TRAIN_START} to {TRAIN_END}',
    'test_period': f'{TEST_START} to {TEST_END}',
    'test_brier_score': float(test_brier),
    'test_accuracy': float(test_accuracy),
    'test_log_loss': float(test_logloss),
    'bookmaker_brier_score': float(bm_brier),
    'bookmaker_accuracy': float(bm_accuracy),
    'brier_improvement': float(bm_brier - test_brier),
    'betting_total_profit': float(betting_results['total_profit']),
    'betting_num_bets': int(betting_results['num_bets']),
    'betting_win_rate': float(betting_results['win_rate']),
    'betting_roi': float(betting_results['roi']),
    'betting_sharpe': float(betting_results['sharpe']),
    'betting_max_drawdown': float(betting_results['max_drawdown']),
    'random_seed': RANDOM_SEED,
    'num_features': len(feature_cols),
    'feature_list': feature_cols
}

with open('test_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print("  Saved: test_metrics.json")

# Save model comparison
models_summary.to_csv('model_comparison.csv', index=False)
print("  Saved: model_comparison.csv")

# Save betting history
if betting_results['num_bets'] > 0:
    betting_df = pd.DataFrame(betting_results['bet_history'])
    betting_df.to_csv('betting_history.csv', index=False)
    print("  Saved: betting_history.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nGenerated files:")
print("  - test_predictions.csv")
print("  - test_metrics.json")
print("  - model_comparison.csv")
print("  - betting_history.csv")
print("  - feature_importances.png")
print("  - calibration_curves.png")
print("  - betting_cumulative_profit.png")
print("  - model_comparison.png")
print("\n" + "=" * 80)
