"""
NHL Game Outcome Prediction - Player-Based Model
Following Holmes & McHale (2024) Approach

This model:
- Calculates individual player ratings based on performance metrics
- Aggregates player ratings to team strength for each game
- Uses time decay to weight recent performance more heavily
- Implements hyperparameter tuning for optimal configuration
- Applies multinomial logistic regression for predictions

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

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Date ranges
TRAIN_START = '2011-01-01'
TRAIN_END = '2020-12-31'
TEST_START = '2021-01-01'
TEST_END = '2023-12-31'

print("=" * 80)
print("NHL PLAYER-BASED PREDICTION MODEL (Holmes & McHale 2024 Approach)")
print("=" * 80)
print(f"Training Period: {TRAIN_START} to {TRAIN_END}")
print(f"Test Period: {TEST_START} to {TEST_END}")
print(f"Random Seed: {RANDOM_SEED}")
print("=" * 80)


# ============================================
# HELPER FUNCTIONS
# ============================================

def calculate_multiclass_brier_score(y_true, y_pred_proba):
    """
    Calculate Brier Score for multi-class classification
    Brier = (1/N) * sum_i sum_k (p_ik - o_ik)^2
    """
    n_samples = len(y_true)
    n_classes = y_pred_proba.shape[1]

    # One-hot encode true labels
    y_true_onehot = np.zeros((n_samples, n_classes))
    y_true_onehot[np.arange(n_samples), y_true] = 1

    # Calculate Brier score
    brier = np.mean(np.sum((y_pred_proba - y_true_onehot) ** 2, axis=1))
    return brier


def calculate_player_ratings(player_df, decay_param=0.001, window_games=None):
    """
    Calculate time-decayed player ratings based on performance metrics

    Parameters:
    - decay_param: time decay parameter (phi) - higher = more weight on recent games
    - window_games: if set, only use last N games for each player

    Returns: Dictionary mapping (playerId, date) -> rating
    """
    print(f"  Calculating player ratings (decay={decay_param}, window={window_games})...")

    # Sort by player and date
    player_df = player_df.sort_values(['playerId', 'date']).reset_index(drop=True)

    # Convert date to datetime if not already
    player_df['date'] = pd.to_datetime(player_df['date'])

    # Initialize rating storage
    player_ratings = {}

    # Process each player separately
    for player_id in player_df['playerId'].unique():
        player_games = player_df[player_df['playerId'] == player_id].copy()

        # Skip players with too few games
        if len(player_games) < 3:
            continue

        # Calculate performance metrics for each game
        # Normalize by time on ice (goals/assists/shots per minute)
        player_games['toi_minutes'] = player_games['toi_seconds'] / 60
        player_games['toi_minutes'] = player_games['toi_minutes'].replace(0, 1)  # Avoid division by zero

        player_games['goals_per_min'] = player_games['goals'] / player_games['toi_minutes']
        player_games['assists_per_min'] = player_games['assists'] / player_games['toi_minutes']
        player_games['shots_per_min'] = player_games['shots'] / player_games['toi_minutes']

        # Plus/minus is already a rating-like measure
        player_games['plus_minus_filled'] = player_games['plusMinus'].fillna(0)

        # Composite performance score
        # Weighted combination: goals are most important, then assists, then shots, then +/-
        player_games['performance'] = (
            3.0 * player_games['goals_per_min'] +
            2.0 * player_games['assists_per_min'] +
            0.5 * player_games['shots_per_min'] +
            0.1 * player_games['plus_minus_filled']
        )

        # Apply windowing if specified
        if window_games is not None:
            player_games = player_games.tail(window_games)

        # Calculate time-decayed ratings for each game
        for idx in range(len(player_games)):
            current_date = player_games.iloc[idx]['date']

            # Get all previous games (before current game)
            past_games = player_games.iloc[:idx]

            if len(past_games) == 0:
                # No history - use neutral rating
                rating = 0.0
            else:
                # Calculate days since each past game
                days_ago = (current_date - past_games['date']).dt.days.values

                # Calculate weights with exponential decay
                weights = np.exp(-decay_param * days_ago)

                # Weighted average of past performances
                performances = past_games['performance'].values
                rating = np.average(performances, weights=weights)

            # Store rating (player, date) -> rating
            player_ratings[(player_id, current_date)] = rating

    return player_ratings


def aggregate_team_strength(game_row, player_ratings, player_df):
    """
    Aggregate player ratings to team strength for a specific game
    """
    game_id = game_row['gameId']
    game_date = game_row['date']
    home_team = game_row['teamname_home']
    away_team = game_row['teamname_away']

    # Get players who played in this game
    game_players = player_df[player_df['gameId'] == game_id]

    # Separate home and away players
    home_players = game_players[game_players['home'] == 1]
    away_players = game_players[game_players['home'] == 0]

    # Aggregate ratings for home team
    home_ratings = []
    for _, player in home_players.iterrows():
        player_id = player['playerId']
        rating = player_ratings.get((player_id, game_date), 0.0)
        home_ratings.append(rating)

    # Aggregate ratings for away team
    away_ratings = []
    for _, player in away_players.iterrows():
        player_id = player['playerId']
        rating = player_ratings.get((player_id, game_date), 0.0)
        away_ratings.append(rating)

    # Team strength = mean of player ratings (could also use sum or median)
    home_strength = np.mean(home_ratings) if len(home_ratings) > 0 else 0.0
    away_strength = np.mean(away_ratings) if len(away_ratings) > 0 else 0.0

    # Also calculate roster size (number of players) as additional feature
    home_roster_size = len(home_ratings)
    away_roster_size = len(away_ratings)

    return home_strength, away_strength, home_roster_size, away_roster_size


# ============================================
# 1. LOAD DATA
# ============================================

print("\n[1/7] Loading data...")

try:
    df = pd.read_csv('se_assignment1_1_data.csv')
    player_df = pd.read_csv('playergamedata.csv')
    print(f"  Game data loaded: {len(df):,} games")
    print(f"  Player data loaded: {len(player_df):,} player-game records")
except FileNotFoundError:
    print("\nERROR: Could not find required data files")
    print("Please ensure 'se_assignment1_1_data.csv' and 'playergamedata.csv' are in the current directory")
    exit(1)

# Convert dates
df['date'] = pd.to_datetime(df['date'])
player_df['date'] = pd.to_datetime(player_df['date'])

# Sort by date
df = df.sort_values('date').reset_index(drop=True)
player_df = player_df.sort_values('date').reset_index(drop=True)

# Remove missing data
critical_cols = ['date', 'result_reg', 'teamname_home', 'teamname_away',
                'team_goals_reg_home', 'team_goals_reg_away',
                'odd_winhome', 'odd_draw', 'odd_awaywin']
df = df.dropna(subset=critical_cols)

print(f"  After removing missing values: {len(df):,} games")

# Convert odds to probabilities
df['bm_prob_home_raw'] = 1 / df['odd_winhome']
df['bm_prob_tie_raw'] = 1 / df['odd_draw']
df['bm_prob_away_raw'] = 1 / df['odd_awaywin']

df['bm_margin'] = df['bm_prob_home_raw'] + df['bm_prob_tie_raw'] + df['bm_prob_away_raw']
df['bm_prob_home'] = df['bm_prob_home_raw'] / df['bm_margin']
df['bm_prob_tie'] = df['bm_prob_tie_raw'] / df['bm_margin']
df['bm_prob_away'] = df['bm_prob_away_raw'] / df['bm_margin']

# Split train/test
train_df = df[(df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)].copy()
test_df = df[(df['date'] >= TEST_START) & (df['date'] <= TEST_END)].copy()

# Also split player data
train_player_df = player_df[(player_df['date'] >= TRAIN_START) & (player_df['date'] <= TRAIN_END)].copy()
test_player_df = player_df[(player_df['date'] >= TEST_START) & (player_df['date'] <= TEST_END)].copy()

print(f"\n  Training games: {len(train_df):,} ({TRAIN_START} to {TRAIN_END})")
print(f"  Test games: {len(test_df):,} ({TEST_START} to {TEST_END})")
print(f"  Training player-games: {len(train_player_df):,}")
print(f"  Test player-games: {len(test_player_df):,}")


# ============================================
# 2. HYPERPARAMETER TUNING
# ============================================

print("\n[2/7] Hyperparameter Tuning on Training/Validation Split...")

# Split training data into train and validation (2011-2018 train, 2019-2020 validate)
train_sub_df = train_df[train_df['date'] < '2019-01-01'].copy()
val_df = train_df[train_df['date'] >= '2019-01-01'].copy()

train_sub_player_df = train_player_df[train_player_df['date'] < '2019-01-01'].copy()
val_player_df = train_player_df[train_player_df['date'] >= '2019-01-01'].copy()

print(f"  Training subset: {len(train_sub_df):,} games (2011-2018)")
print(f"  Validation set: {len(val_df):,} games (2019-2020)")

# Define hyperparameter grid
decay_params = [0.0001, 0.0005, 0.001, 0.002]  # Time decay (phi)
window_games_list = [None, 50, 100, 200]  # Rolling window (None = all history)

print(f"\n  Testing {len(decay_params)} x {len(window_games_list)} = {len(decay_params) * len(window_games_list)} configurations...")

best_brier = float('inf')
best_config = None
results = []

for decay in decay_params:
    for window in window_games_list:
        print(f"\n  Testing decay={decay}, window={window}...")

        # Calculate player ratings on training data
        player_ratings = calculate_player_ratings(
            train_sub_player_df,
            decay_param=decay,
            window_games=window
        )

        # Build features for training subset
        print("    Building training features...")
        train_features = []
        for idx, row in train_sub_df.iterrows():
            home_str, away_str, home_size, away_size = aggregate_team_strength(
                row, player_ratings, train_sub_player_df
            )
            train_features.append([
                home_str - away_str,  # Strength difference
                home_str,              # Home team strength
                away_str,              # Away team strength
                home_size - away_size  # Roster size difference
            ])

        X_train_sub = np.array(train_features)
        y_train_sub = train_sub_df['result_reg'].values

        # Handle NaN values (replace with 0)
        X_train_sub = np.nan_to_num(X_train_sub, nan=0.0)

        # Build features for validation set
        # IMPORTANT: Update player ratings with all games up to validation
        print("    Building validation features...")
        all_train_player_df = pd.concat([train_sub_player_df, val_player_df])
        player_ratings_full = calculate_player_ratings(
            all_train_player_df,
            decay_param=decay,
            window_games=window
        )

        val_features = []
        for idx, row in val_df.iterrows():
            home_str, away_str, home_size, away_size = aggregate_team_strength(
                row, player_ratings_full, all_train_player_df
            )
            val_features.append([
                home_str - away_str,
                home_str,
                away_str,
                home_size - away_size
            ])

        X_val = np.array(val_features)
        y_val = val_df['result_reg'].values

        # Handle NaN values
        X_val = np.nan_to_num(X_val, nan=0.0)

        # Train model
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

        # Evaluate on validation set
        val_pred_proba = model.predict_proba(X_val_scaled)
        val_brier = calculate_multiclass_brier_score(y_val, val_pred_proba)
        val_accuracy = accuracy_score(y_val, model.predict(X_val_scaled))

        print(f"    Validation Brier: {val_brier:.4f}, Accuracy: {val_accuracy:.4f}")

        results.append({
            'decay': decay,
            'window': window,
            'val_brier': val_brier,
            'val_accuracy': val_accuracy
        })

        # Track best configuration
        if val_brier < best_brier:
            best_brier = val_brier
            best_config = {'decay': decay, 'window': window}
            print(f"    >>> NEW BEST! Brier: {val_brier:.4f}")

print(f"\n  Best configuration: decay={best_config['decay']}, window={best_config['window']}")
print(f"  Best validation Brier: {best_brier:.4f}")

# Save hyperparameter tuning results
tuning_df = pd.DataFrame(results)
tuning_df.to_csv('hyperparameter_tuning_results.csv', index=False)
print(f"\n  Hyperparameter tuning results saved to: hyperparameter_tuning_results.csv")


# ============================================
# 3. RETRAIN ON FULL TRAINING SET
# ============================================

print("\n[3/7] Retraining with best hyperparameters on full training set (2011-2020)...")

# Calculate player ratings with best config on ALL training data
final_player_ratings = calculate_player_ratings(
    train_player_df,
    decay_param=best_config['decay'],
    window_games=best_config['window']
)

# Build training features
print("  Building training features...")
train_features_final = []
for idx, row in train_df.iterrows():
    home_str, away_str, home_size, away_size = aggregate_team_strength(
        row, final_player_ratings, train_player_df
    )
    train_features_final.append([
        home_str - away_str,
        home_str,
        away_str,
        home_size - away_size
    ])

X_train_final = np.array(train_features_final)
y_train_final = train_df['result_reg'].values

# Handle NaN values
X_train_final = np.nan_to_num(X_train_final, nan=0.0)

# Scale and train
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

print("  MODEL FROZEN - ready for test evaluation")


# ============================================
# 4. BOOKMAKER BENCHMARK
# ============================================

print("\n[4/7] Calculating bookmaker benchmark on test set...")

bm_probs_test = test_df[['bm_prob_away', 'bm_prob_tie', 'bm_prob_home']].values
y_test = test_df['result_reg'].values

bm_brier = calculate_multiclass_brier_score(y_test, bm_probs_test)
bm_pred = np.argmax(bm_probs_test, axis=1)
bm_accuracy = accuracy_score(y_test, bm_pred)
bm_logloss = log_loss(y_test, bm_probs_test)

print(f"  Bookmaker Brier Score: {bm_brier:.4f}")
print(f"  Bookmaker Accuracy: {bm_accuracy:.4f}")
print(f"  Bookmaker Log Loss: {bm_logloss:.4f}")


# ============================================
# 5. TEST SET EVALUATION
# ============================================

print("\n[5/7] Evaluating on TEST SET (2021-2023)...")

# IMPORTANT: Update player ratings through test period (following fixed update rule)
print("  Updating player ratings through test period...")
all_player_df = pd.concat([train_player_df, test_player_df])
test_player_ratings = calculate_player_ratings(
    all_player_df,
    decay_param=best_config['decay'],
    window_games=best_config['window']
)

# Build test features
print("  Building test features...")
test_features = []
for idx, row in test_df.iterrows():
    home_str, away_str, home_size, away_size = aggregate_team_strength(
        row, test_player_ratings, all_player_df
    )
    test_features.append([
        home_str - away_str,
        home_str,
        away_str,
        home_size - away_size
    ])

X_test = np.array(test_features)

# Handle NaN values
X_test = np.nan_to_num(X_test, nan=0.0)

X_test_scaled = scaler_final.transform(X_test)

# Generate predictions
test_pred_proba = final_model.predict_proba(X_test_scaled)
test_pred = final_model.predict(X_test_scaled)

# Calculate metrics
test_brier = calculate_multiclass_brier_score(y_test, test_pred_proba)
test_accuracy = accuracy_score(y_test, test_pred)
test_logloss = log_loss(y_test, test_pred_proba)

print("\n" + "=" * 80)
print("FINAL TEST SET RESULTS (2021-2023)")
print("=" * 80)
print(f"Model: Player-Based Ratings + Multinomial Logistic Regression")
print(f"Best hyperparameters: decay={best_config['decay']}, window={best_config['window']}")
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
# 6. BETTING SIMULATION
# ============================================

print("\n[6/7] Running betting simulation on TEST SET...")

EV_THRESHOLD = 0.03
FLAT_STAKE = 10

bets = []
total_profit = 0
total_staked = 0

for idx in range(len(test_df)):
    row = test_df.iloc[idx]
    pred_probs = test_pred_proba[idx]
    actual = y_test[idx]

    # Bookmaker probabilities
    bm_prob = np.array([row['bm_prob_away'], row['bm_prob_tie'], row['bm_prob_home']])

    # Calculate edges
    edges = pred_probs - bm_prob

    # Bet on outcome with largest positive edge (if above threshold)
    max_edge_idx = np.argmax(edges)
    max_edge = edges[max_edge_idx]

    if max_edge >= EV_THRESHOLD:
        # Place bet
        stake = FLAT_STAKE
        total_staked += stake

        # Check if bet wins
        if max_edge_idx == actual:
            # Win - get back stake plus profit
            if max_edge_idx == 0:  # Away
                payout = stake * row['odd_awaywin']
            elif max_edge_idx == 1:  # Tie
                payout = stake * row['odd_draw']
            else:  # Home
                payout = stake * row['odd_winhome']

            profit = payout - stake
        else:
            # Lose
            profit = -stake

        total_profit += profit

        bets.append({
            'gameId': row['gameId'],
            'date': row['date'],
            'bet_on': max_edge_idx,
            'actual': actual,
            'edge': max_edge,
            'stake': stake,
            'profit': profit,
            'won': int(max_edge_idx == actual)
        })

# Calculate metrics
num_bets = len(bets)
num_wins = sum([b['won'] for b in bets])
win_rate = num_wins / num_bets if num_bets > 0 else 0
roi = (total_profit / total_staked) if total_staked > 0 else 0

# Calculate Sharpe ratio
if num_bets > 1:
    bet_returns = [b['profit'] / b['stake'] for b in bets]
    sharpe = (np.mean(bet_returns) / np.std(bet_returns)) * np.sqrt(num_bets) if np.std(bet_returns) > 0 else 0
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
print(f"Win Rate:          {win_rate*100:.2f}%")
print(f"ROI:               {roi*100:.2f}%")
print(f"Sharpe Ratio:      {sharpe:.4f}")
print(f"Max Drawdown:      ${max_drawdown:.2f}")
print("=" * 80)


# ============================================
# 7. SAVE RESULTS
# ============================================

print("\n[7/7] Saving results...")

# Save predictions
test_df['pred_prob_away'] = test_pred_proba[:, 0]
test_df['pred_prob_tie'] = test_pred_proba[:, 1]
test_df['pred_prob_home'] = test_pred_proba[:, 2]
test_df['predicted_outcome'] = test_pred

test_df.to_csv('test_predictions_player_based.csv', index=False)
print("  Saved: test_predictions_player_based.csv")

# Save metrics
metrics = {
    'model_name': 'Player-Based Ratings + Multinomial Logistic Regression',
    'approach': 'Holmes & McHale (2024)',
    'hyperparameters': {
        'decay_param': best_config['decay'],
        'window_games': best_config['window'],
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

with open('test_metrics_player_based.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("  Saved: test_metrics_player_based.json")

# Save betting history
if len(bets) > 0:
    bets_df = pd.DataFrame(bets)
    bets_df.to_csv('betting_history_player_based.csv', index=False)
    print("  Saved: betting_history_player_based.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nPlayer-based model following Holmes & McHale (2024):")
print("  - Individual player ratings based on performance metrics")
print("  - Time-decayed weighting of past games")
print("  - Hyperparameter tuning for optimal configuration")
print("  - Aggregates player ratings to team strength")
print("  - Multinomial logistic regression for predictions")
print("=" * 80)
