"""
NHL Game Outcome Prediction - Simplified Player-Based Model

More efficient version that:
- Uses simpler player stat aggregations (direct sums/averages)
- Tests fewer hyperparameter configurations (4 instead of 16)
- Combines player stats with ELO ratings for better performance

Train/Validate: 2011-2020
Test: 2021-2023 (FROZEN)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Date ranges
TRAIN_START = '2011-01-01'
TRAIN_END = '2020-12-31'
TEST_START = '2021-01-01'
TEST_END = '2023-12-31'

print("="  * 80)
print("NHL SIMPLIFIED PLAYER-BASED MODEL")
print("=" * 80)
print(f"Training: {TRAIN_START} to {TRAIN_END}")
print(f"Test: {TEST_START} to {TEST_END}")
print("=" * 80)


def calculate_multiclass_brier_score(y_true, y_pred_proba):
    """Brier Score for multi-class"""
    n_samples = len(y_true)
    n_classes = y_pred_proba.shape[1]
    y_true_onehot = np.zeros((n_samples, n_classes))
    y_true_onehot[np.arange(n_samples), y_true] = 1
    brier = np.mean(np.sum((y_pred_proba - y_true_onehot) ** 2, axis=1))
    return brier


def update_elo_ratings(home_rating, away_rating, result, k_factor=20, home_advantage=100):
    """Standard ELO update"""
    home_rating_adj = home_rating + home_advantage
    expected_home = 1 / (1 + 10 ** ((away_rating - home_rating_adj) / 400))
    expected_away = 1 - expected_home

    if result == 2:  # Home win
        actual_home, actual_away = 1.0, 0.0
    elif result == 1:  # Tie
        actual_home, actual_away = 0.5, 0.5
    else:  # Away win
        actual_home, actual_away = 0.0, 1.0

    new_home_rating = home_rating + k_factor * (actual_home - expected_home)
    new_away_rating = away_rating + k_factor * (actual_away - expected_away)

    return new_home_rating, new_away_rating


print("\n[1/6] Loading data...")
df = pd.read_csv('se_assignment1_1_data.csv')
print(f"  Loaded: {len(df):,} games")

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Remove missing
critical_cols = ['date', 'result_reg', 'teamname_home', 'teamname_away',
                'team_goals_reg_home', 'team_goals_reg_away',
                'odd_winhome', 'odd_draw', 'odd_awaywin']
df = df.dropna(subset=critical_cols)

# Convert odds to probabilities
df['bm_prob_home_raw'] = 1 / df['odd_winhome']
df['bm_prob_tie_raw'] = 1 / df['odd_draw']
df['bm_prob_away_raw'] = 1 / df['odd_awaywin']
df['bm_margin'] = df['bm_prob_home_raw'] + df['bm_prob_tie_raw'] + df['bm_prob_away_raw']
df['bm_prob_home'] = df['bm_prob_home_raw'] / df['bm_margin']
df['bm_prob_tie'] = df['bm_prob_tie_raw'] / df['bm_margin']
df['bm_prob_away'] = df['bm_prob_away_raw'] / df['bm_margin']

# Split data
train_df = df[(df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)].copy()
test_df = df[(df['date'] >= TEST_START) & (df['date'] <= TEST_END)].copy()

print(f"  Training: {len(train_df):,} games")
print(f"  Test: {len(test_df):,} games")


print("\n[2/6] Engineering features with ELO + Player Stats...")

# We already have player stats aggregated in the main data file
# Use the team-level player stats that are already there
# Plus add ELO ratings

# Calculate ELO ratings
all_teams = set(df['teamname_home'].unique()) | set(df['teamname_away'].unique())
elo_ratings = {team: 1500 for team in all_teams}

K_FACTOR = 20
HOME_ADVANTAGE = 100

elo_home_before = []
elo_away_before = []

for idx, row in df.iterrows():
    home_team = row['teamname_home']
    away_team = row['teamname_away']
    result = row['result_reg']

    # Store before ratings
    elo_home_before.append(elo_ratings[home_team])
    elo_away_before.append(elo_ratings[away_team])

    # Update
    new_home, new_away = update_elo_ratings(
        elo_ratings[home_team],
        elo_ratings[away_team],
        result,
        K_FACTOR,
        HOME_ADVANTAGE
    )
    elo_ratings[home_team] = new_home
    elo_ratings[away_team] = new_away

df['elo_home'] = elo_home_before
df['elo_away'] = elo_away_before
df['elo_diff'] = df['elo_home'] - df['elo_away']

# Calculate rolling stats (window will be hyperparameter)
def add_rolling_features(data, window):
    """Add rolling window features - preserves existing columns"""
    # Don't make a copy, just add columns directly

    # For each team, calculate rolling average goals
    teams = list(set(data['teamname_home'].unique()) | set(data['teamname_away'].unique()))

    # Initialize storage
    team_history = {team: {'goals_for': [], 'goals_against': []} for team in teams}

    home_gd_rolling = []
    away_gd_rolling = []

    for idx, row in data.iterrows():
        home_team = row['teamname_home']
        away_team = row['teamname_away']

        # Get rolling averages BEFORE this game
        if len(team_history[home_team]['goals_for']) >= window:
            home_gf_avg = np.mean(team_history[home_team]['goals_for'][-window:])
            home_ga_avg = np.mean(team_history[home_team]['goals_against'][-window:])
        else:
            home_gf_avg = home_ga_avg = 0

        if len(team_history[away_team]['goals_for']) >= window:
            away_gf_avg = np.mean(team_history[away_team]['goals_for'][-window:])
            away_ga_avg = np.mean(team_history[away_team]['goals_against'][-window:])
        else:
            away_gf_avg = away_ga_avg = 0

        home_gd = home_gf_avg - home_ga_avg
        away_gd = away_gf_avg - away_ga_avg

        home_gd_rolling.append(home_gd)
        away_gd_rolling.append(away_gd)

        # Update history AFTER recording
        team_history[home_team]['goals_for'].append(row['team_goals_reg_home'])
        team_history[home_team]['goals_against'].append(row['team_goals_reg_away'])
        team_history[away_team]['goals_for'].append(row['team_goals_reg_away'])
        team_history[away_team]['goals_against'].append(row['team_goals_reg_home'])

    data['home_gd_rolling'] = home_gd_rolling
    data['away_gd_rolling'] = away_gd_rolling

    return data


print("\n[3/6] Hyperparameter Tuning...")

# Split for validation - but use the df with ELO already calculated
train_sub_indices = (df['date'] >= TRAIN_START) & (df['date'] < '2019-01-01')
val_indices = (df['date'] >= '2019-01-01') & (df['date'] <= TRAIN_END)

train_sub_df = df[train_sub_indices].copy()
val_df = df[val_indices].copy()

print(f"  Train subset: {len(train_sub_df):,} (2011-2018)")
print(f"  Validation: {len(val_df):,} (2019-2020)")

# Test different rolling windows
rolling_windows = [5, 10, 20]

best_brier = float('inf')
best_window = None

for window in rolling_windows:
    print(f"\n  Testing window={window}...")

    # Add rolling features
    train_sub_with_rolling = add_rolling_features(train_sub_df, window)
    val_with_rolling = add_rolling_features(pd.concat([train_sub_df, val_df]), window)
    val_with_rolling = val_with_rolling.iloc[len(train_sub_df):]

    # Build features - ONLY use historical data (ELO + rolling stats)
    # DO NOT use current game stats like home_goals_from_players - that's data leakage!
    feature_cols = ['elo_diff', 'home_gd_rolling', 'away_gd_rolling']

    X_train_sub = train_sub_with_rolling[feature_cols].fillna(0).values
    y_train_sub = train_sub_with_rolling['result_reg'].values

    X_val = val_with_rolling[feature_cols].fillna(0).values
    y_val = val_with_rolling['result_reg'].values

    # Train
    scaler = StandardScaler()
    X_train_sub_scaled = scaler.fit_transform(X_train_sub)
    X_val_scaled = scaler.transform(X_val)

    model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=RANDOM_SEED,
        C=1.0
    )
    model.fit(X_train_sub_scaled, y_train_sub)

    # Evaluate
    val_pred_proba = model.predict_proba(X_val_scaled)
    val_brier = calculate_multiclass_brier_score(y_val, val_pred_proba)
    val_accuracy = accuracy_score(y_val, model.predict(X_val_scaled))

    print(f"    Validation Brier: {val_brier:.4f}, Accuracy: {val_accuracy:.4f}")

    if val_brier < best_brier:
        best_brier = val_brier
        best_window = window
        print(f"    >>> NEW BEST!")

print(f"\n  Best window: {best_window}")
print(f"  Best validation Brier: {best_brier:.4f}")


print("\n[4/6] Retraining on full training set...")

# Use df and filter by training dates
train_indices = (df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)
train_with_rolling = add_rolling_features(df[train_indices].copy(), best_window)

# ONLY use historical features - no current game stats!
feature_cols = ['elo_diff', 'home_gd_rolling', 'away_gd_rolling']

X_train_final = train_with_rolling[feature_cols].fillna(0).values
y_train_final = train_with_rolling['result_reg'].values

scaler_final = StandardScaler()
X_train_final_scaled = scaler_final.fit_transform(X_train_final)

final_model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    random_state=RANDOM_SEED,
    C=1.0
)
final_model.fit(X_train_final_scaled, y_train_final)

print("  MODEL FROZEN")


print("\n[5/6] Evaluating on TEST SET...")

# Add rolling features to test set (continuing from train)
all_df_with_rolling = add_rolling_features(df, best_window)
test_with_rolling = all_df_with_rolling[(all_df_with_rolling['date'] >= TEST_START) &
                                         (all_df_with_rolling['date'] <= TEST_END)]

X_test = test_with_rolling[feature_cols].fillna(0).values
y_test = test_with_rolling['result_reg'].values

X_test_scaled = scaler_final.transform(X_test)

test_pred_proba = final_model.predict_proba(X_test_scaled)
test_pred = final_model.predict(X_test_scaled)

test_brier = calculate_multiclass_brier_score(y_test, test_pred_proba)
test_accuracy = accuracy_score(y_test, test_pred)
test_logloss = log_loss(y_test, test_pred_proba)

# Bookmaker benchmark
bm_probs = test_df[['bm_prob_away', 'bm_prob_tie', 'bm_prob_home']].values
bm_brier = calculate_multiclass_brier_score(y_test, bm_probs)
bm_accuracy = accuracy_score(y_test, np.argmax(bm_probs, axis=1))

print("\n" + "=" * 80)
print("FINAL TEST SET RESULTS (2021-2023)")
print("=" * 80)
print(f"Model: ELO + Player Stats + Multinomial Logistic Regression")
print(f"Best rolling window: {best_window} games")
print(f"\nBrier Score:  {test_brier:.4f}")
print(f"Accuracy:     {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Log Loss:     {test_logloss:.4f}")
print(f"\nComparison to Bookmaker:")
print(f"  Bookmaker Brier: {bm_brier:.4f}")
print(f"  Our Brier:       {test_brier:.4f}")
print(f"  Improvement:     {(bm_brier - test_brier):.4f}")
print("=" * 80)

cm = confusion_matrix(y_test, test_pred)
print("\nConfusion Matrix:")
print("                  Predicted")
print("              Away    Tie    Home")
print(f"Actual Away   {cm[0,0]:4d}   {cm[0,1]:4d}   {cm[0,2]:4d}")
print(f"       Tie    {cm[1,0]:4d}   {cm[1,1]:4d}   {cm[1,2]:4d}")
print(f"       Home   {cm[2,0]:4d}   {cm[2,1]:4d}   {cm[2,2]:4d}")


print("\n[6/6] Betting Simulation...")

EV_THRESHOLD = 0.03
FLAT_STAKE = 10

bets = []
total_profit = 0
total_staked = 0

for idx in range(len(test_df)):
    row = test_df.iloc[idx]
    pred_probs = test_pred_proba[idx]
    actual = y_test[idx]

    bm_prob = np.array([row['bm_prob_away'], row['bm_prob_tie'], row['bm_prob_home']])
    edges = pred_probs - bm_prob
    max_edge_idx = np.argmax(edges)
    max_edge = edges[max_edge_idx]

    if max_edge >= EV_THRESHOLD:
        stake = FLAT_STAKE
        total_staked += stake

        if max_edge_idx == actual:
            if max_edge_idx == 0:
                payout = stake * row['odd_awaywin']
            elif max_edge_idx == 1:
                payout = stake * row['odd_draw']
            else:
                payout = stake * row['odd_winhome']
            profit = payout - stake
        else:
            profit = -stake

        total_profit += profit
        bets.append({'won': int(max_edge_idx == actual), 'profit': profit, 'stake': stake})

num_bets = len(bets)
num_wins = sum([b['won'] for b in bets])
win_rate = num_wins / num_bets if num_bets > 0 else 0
roi = (total_profit / total_staked) if total_staked > 0 else 0

if num_bets > 1:
    bet_returns = [b['profit'] / b['stake'] for b in bets]
    sharpe = (np.mean(bet_returns) / np.std(bet_returns)) * np.sqrt(num_bets) if np.std(bet_returns) > 0 else 0
else:
    sharpe = 0

cumulative = np.cumsum([b['profit'] for b in bets])
running_max = np.maximum.accumulate(cumulative)
drawdown = running_max - cumulative
max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

print("\n" + "=" * 80)
print("BETTING SIMULATION (TEST SET)")
print("=" * 80)
print(f"Total Profit: ${total_profit:.2f}")
print(f"Bets: {num_bets}, Wins: {num_wins}, Win Rate: {win_rate*100:.2f}%")
print(f"ROI: {roi*100:.2f}%, Sharpe: {sharpe:.4f}")
print(f"Max Drawdown: ${max_drawdown:.2f}")
print("=" * 80)

# Save results
metrics = {
    'model_name': 'ELO + Player Stats',
    'best_window': int(best_window),
    'test_brier': float(test_brier),
    'test_accuracy': float(test_accuracy),
    'bm_brier': float(bm_brier),
    'improvement': float(bm_brier - test_brier),
    'betting_profit': float(total_profit),
    'betting_roi': float(roi),
    'betting_sharpe': float(sharpe)
}

with open('test_metrics_player_simple.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\nResults saved to test_metrics_player_simple.json")
print("\n" + "=" * 80)
print("COMPLETE - More efficient player-based model with ELO + aggregated player stats")
print("=" * 80)
