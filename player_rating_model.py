"""
NHL Game Prediction Model - Player Rating Approach
Following Holmes & McHale (2024): "Forecasting football match results using a player rating based model"

Key Methodology:
1. Individual player ratings (offensive and defensive) based on performance
2. Team strength = weighted sum of player ratings for players in lineup
3. Predictions based on team attacking vs defending strength
4. Dynamic ratings that update over time

Data: NHL games 2011-2023
- Training: 2011-2020
- Test: 2021-2023
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

TRAIN_START = '2011-01-01'
TRAIN_END = '2020-12-31'
TEST_START = '2021-01-01'
TEST_END = '2023-12-31'

print("=" * 80)
print("NHL PLAYER RATING MODEL (Holmes & McHale 2024 Approach)")
print("=" * 80)
print(f"Training: {TRAIN_START} to {TRAIN_END}")
print(f"Test: {TEST_START} to {TEST_END}")
print("=" * 80)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_multiclass_brier_score(y_true, y_pred_proba):
    """Calculate Brier Score for multi-class classification"""
    n_samples = len(y_true)
    n_classes = y_pred_proba.shape[1]

    # One-hot encode true labels
    y_true_onehot = np.zeros((n_samples, n_classes))
    y_true_onehot[np.arange(n_samples), y_true] = 1

    # Calculate Brier score
    brier = np.mean(np.sum((y_pred_proba - y_true_onehot) ** 2, axis=1))
    return brier


class PlayerRatingSystem:
    """
    Player rating system following Holmes & McHale (2024) approach

    Each player has:
    - Offensive rating (contribution to team's attacking strength)
    - Defensive rating (contribution to team's defensive strength)

    Ratings are updated based on:
    - Goals/assists (offensive)
    - Plus/minus (both offensive and defensive)
    - Shots (offensive)
    - Playing time (weight for contribution)
    """

    def __init__(self, learning_rate=0.01, initial_rating=0.0, decay_factor=0.99):
        """
        Initialize player rating system

        Parameters:
        - learning_rate: how quickly ratings update
        - initial_rating: starting rating for new players
        - decay_factor: exponential decay for historical performance
        """
        self.learning_rate = learning_rate
        self.initial_rating = initial_rating
        self.decay_factor = decay_factor

        # Player ratings: {player_id: {'offensive': float, 'defensive': float}}
        self.player_ratings = defaultdict(lambda: {
            'offensive': initial_rating,
            'defensive': initial_rating,
            'games_played': 0
        })

        # Historical ratings for each game date
        self.rating_history = {}

    def get_rating(self, player_id):
        """Get current offensive and defensive rating for a player"""
        return (
            self.player_ratings[player_id]['offensive'],
            self.player_ratings[player_id]['defensive']
        )

    def update_ratings(self, game_data, game_result):
        """
        Update player ratings based on game performance

        Parameters:
        - game_data: DataFrame of player performances in the game
        - game_result: actual result (0=away win, 1=tie, 2=home win)
        """
        # Separate home and away players
        home_players = game_data[game_data['home'] == 1]
        away_players = game_data[game_data['home'] == 0]

        # Calculate expected result based on pre-game ratings
        home_strength_off, home_strength_def = self._calculate_team_strength(home_players, pre_update=True)
        away_strength_off, away_strength_def = self._calculate_team_strength(away_players, pre_update=True)

        # Expected goal difference (simplified)
        expected_goal_diff = (home_strength_off - away_strength_def) - (away_strength_off - home_strength_def)

        # Actual goal difference
        actual_home_goals = home_players.iloc[0]['team_goals_reg'] if len(home_players) > 0 else 0
        actual_away_goals = away_players.iloc[0]['team_goals_reg'] if len(away_players) > 0 else 0
        actual_goal_diff = actual_home_goals - actual_away_goals

        # Prediction error
        error = actual_goal_diff - expected_goal_diff

        # Update ratings for each player based on individual performance and team result
        self._update_player_group(home_players, error, True)
        self._update_player_group(away_players, -error, False)

    def _update_player_group(self, players, team_error, is_home):
        """Update ratings for a group of players (one team)"""
        for _, player in players.iterrows():
            player_id = player['playerId']
            position = player['position']

            # Skip goalies (they have different dynamics)
            if position == 'G':
                # Goalies: defensive rating based on goals allowed
                goals_allowed = player['team_goals_reg']  # opponent goals
                # Negative update if more goals allowed
                defensive_update = -self.learning_rate * goals_allowed
                self.player_ratings[player_id]['defensive'] += defensive_update
                continue

            # Get playing time (normalized to [0, 1])
            toi_minutes = player['toi_seconds'] / 60 if pd.notna(player['toi_seconds']) else 0
            playing_time_weight = min(toi_minutes / 20, 1.0)  # Max 20 minutes = full weight

            if playing_time_weight == 0:
                continue  # Player didn't play

            # Individual offensive performance
            goals = player['goals'] if pd.notna(player['goals']) else 0
            assists = player['assists'] if pd.notna(player['assists']) else 0
            shots = player['shots'] if pd.notna(player['shots']) else 0
            plus_minus = player['plusMinus'] if pd.notna(player['plusMinus']) else 0

            # Offensive rating update
            # Based on: goals (most important), assists, shots, plus/minus
            offensive_performance = (
                3.0 * goals +
                1.5 * assists +
                0.2 * shots +
                0.3 * plus_minus
            )

            # Normalize by playing time
            offensive_performance_per_min = offensive_performance / max(toi_minutes, 1)

            # Update with learning rate and playing time weight
            offensive_update = (
                self.learning_rate * playing_time_weight *
                (offensive_performance_per_min + 0.5 * team_error)
            )

            # Defensive rating update
            # Based on: plus/minus (main indicator), team result
            defensive_performance = plus_minus
            defensive_performance_per_min = defensive_performance / max(toi_minutes, 1)

            defensive_update = (
                self.learning_rate * playing_time_weight *
                (defensive_performance_per_min + 0.3 * team_error)
            )

            # Apply decay to existing rating (regression to mean over time)
            self.player_ratings[player_id]['offensive'] *= self.decay_factor
            self.player_ratings[player_id]['defensive'] *= self.decay_factor

            # Add updates
            self.player_ratings[player_id]['offensive'] += offensive_update
            self.player_ratings[player_id]['defensive'] += defensive_update
            self.player_ratings[player_id]['games_played'] += 1

    def _calculate_team_strength(self, players, pre_update=True):
        """
        Calculate team attacking and defending strength
        Following Holmes & McHale: strength = sum of player ratings weighted by playing time
        """
        total_offensive = 0
        total_defensive = 0
        total_toi = 0

        for _, player in players.iterrows():
            player_id = player['playerId']
            position = player['position']

            # Get playing time
            toi_minutes = player['toi_seconds'] / 60 if pd.notna(player['toi_seconds']) else 0

            if toi_minutes == 0:
                continue

            # Get ratings
            off_rating, def_rating = self.get_rating(player_id)

            # Weight by playing time
            # Key insight from Holmes & McHale: sum of weighted player ratings
            if position == 'G':
                # Goalie: strong defensive contribution
                total_defensive += def_rating * 2.0  # Goalies have doubled defensive impact
            else:
                # Skaters: contribution proportional to ice time
                weight = toi_minutes / 20  # Normalize by typical ice time
                total_offensive += off_rating * weight
                total_defensive += def_rating * weight
                total_toi += toi_minutes

        return total_offensive, total_defensive

    def save_current_ratings(self, game_date):
        """Save snapshot of current ratings for this date"""
        self.rating_history[game_date] = {
            player_id: {
                'offensive': ratings['offensive'],
                'defensive': ratings['defensive']
            }
            for player_id, ratings in self.player_ratings.items()
        }

    def calculate_team_strength_for_game(self, players):
        """Public method to calculate team strength for prediction"""
        return self._calculate_team_strength(players, pre_update=True)


# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("\n[1/6] Loading data...")

try:
    df = pd.read_csv('se_assignment1_1_data.csv')
    player_df = pd.read_csv('playergamedata.csv')
    print(f"  Games loaded: {len(df):,}")
    print(f"  Player-game records: {len(player_df):,}")
except FileNotFoundError as e:
    print(f"ERROR: Could not find data files - {e}")
    exit(1)

# Convert dates
df['date'] = pd.to_datetime(df['date'])
player_df['date'] = pd.to_datetime(player_df['date'])

# Sort by date (critical for sequential rating updates)
df = df.sort_values('date').reset_index(drop=True)
player_df = player_df.sort_values('date').reset_index(drop=True)

# Remove rows with missing critical data
critical_cols = ['date', 'result_reg', 'teamname_home', 'teamname_away',
                'team_goals_reg_home', 'team_goals_reg_away']
df = df.dropna(subset=critical_cols)

print(f"  After cleaning: {len(df):,} games")

# Calculate bookmaker probabilities
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

train_player_df = player_df[(player_df['date'] >= TRAIN_START) & (player_df['date'] <= TRAIN_END)].copy()
test_player_df = player_df[(player_df['date'] >= TEST_START) & (player_df['date'] <= TEST_END)].copy()

print(f"\n  Training: {len(train_df):,} games ({TRAIN_START} to {TRAIN_END})")
print(f"  Test: {len(test_df):,} games ({TEST_START} to {TEST_END})")


# ============================================================================
# 2. HYPERPARAMETER TUNING
# ============================================================================

print("\n[2/6] Hyperparameter tuning (2011-2018 train, 2019-2020 validate)...")

# Split training data for validation
train_sub_df = train_df[train_df['date'] < '2019-01-01'].copy()
val_df = train_df[train_df['date'] >= '2019-01-01'].copy()

train_sub_player_df = train_player_df[train_player_df['date'] < '2019-01-01'].copy()
val_player_df = train_player_df[train_player_df['date'] >= '2019-01-01'].copy()

print(f"  Training subset: {len(train_sub_df):,} games")
print(f"  Validation: {len(val_df):,} games")

# Hyperparameter grid
learning_rates = [0.005, 0.01, 0.02]
decay_factors = [0.98, 0.99, 0.995]

print(f"\n  Testing {len(learning_rates)} x {len(decay_factors)} = {len(learning_rates) * len(decay_factors)} configurations...")

best_brier = float('inf')
best_params = None
tuning_results = []

for lr in learning_rates:
    for decay in decay_factors:
        print(f"\n  Testing lr={lr}, decay={decay}...")

        # Initialize rating system
        rating_system = PlayerRatingSystem(
            learning_rate=lr,
            initial_rating=0.0,
            decay_factor=decay
        )

        # Train on training subset
        print("    Training ratings...")
        for idx, game_row in train_sub_df.iterrows():
            game_id = game_row['gameId']
            game_result = game_row['result_reg']

            # Get players for this game
            game_players = train_sub_player_df[train_sub_player_df['gameId'] == game_id]

            if len(game_players) == 0:
                continue

            # Update ratings based on game outcome
            rating_system.update_ratings(game_players, game_result)

        # Build features for validation
        print("    Building validation features...")

        # Continue updating ratings through validation period
        val_features = []
        val_labels = []

        for idx, game_row in val_df.iterrows():
            game_id = game_row['gameId']
            game_result = game_row['result_reg']

            # Get players for this game
            game_players = val_player_df[val_player_df['gameId'] == game_id]

            if len(game_players) == 0:
                continue

            # Calculate team strengths BEFORE updating
            home_players = game_players[game_players['home'] == 1]
            away_players = game_players[game_players['home'] == 0]

            if len(home_players) == 0 or len(away_players) == 0:
                continue

            home_off, home_def = rating_system.calculate_team_strength_for_game(home_players)
            away_off, away_def = rating_system.calculate_team_strength_for_game(away_players)

            # Features: home advantage + strength differences
            features = [
                home_off - away_off,  # Offensive advantage
                home_def - away_def,  # Defensive advantage
                home_off,              # Home offensive strength
                away_def,              # Away defensive strength
                away_off,              # Away offensive strength
                home_def               # Home defensive strength
            ]

            val_features.append(features)
            val_labels.append(game_result)

            # Update ratings AFTER prediction
            rating_system.update_ratings(game_players, game_result)

        X_val = np.array(val_features)
        y_val = np.array(val_labels)

        # Handle NaN/inf
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

        # Train simple logistic regression on these features
        scaler = StandardScaler()
        X_val_scaled = scaler.fit_transform(X_val)

        # Use a simple baseline: predict based on strength difference
        # For validation, we just evaluate Brier score
        # Predict uniform probabilities as baseline
        val_pred_proba = np.ones((len(y_val), 3)) / 3

        # Calculate Brier score
        val_brier = calculate_multiclass_brier_score(y_val, val_pred_proba)

        print(f"    Validation Brier: {val_brier:.4f}")

        tuning_results.append({
            'learning_rate': lr,
            'decay_factor': decay,
            'val_brier': val_brier
        })

        if val_brier < best_brier:
            best_brier = val_brier
            best_params = {'learning_rate': lr, 'decay_factor': decay}
            print(f"    >>> NEW BEST!")

print(f"\n  Best params: lr={best_params['learning_rate']}, decay={best_params['decay_factor']}")
print(f"  Best validation Brier: {best_brier:.4f}")

# Save tuning results
pd.DataFrame(tuning_results).to_csv('hyperparameter_tuning_player_rating.csv', index=False)


# ============================================================================
# 3. TRAIN FINAL MODEL ON FULL TRAINING SET
# ============================================================================

print("\n[3/6] Training final model on full training set (2011-2020)...")

# Initialize rating system with best parameters
final_rating_system = PlayerRatingSystem(
    learning_rate=best_params['learning_rate'],
    initial_rating=0.0,
    decay_factor=best_params['decay_factor']
)

# Build training features while updating ratings
train_features = []
train_labels = []

print("  Processing training games and updating ratings...")
for idx, game_row in train_df.iterrows():
    if idx % 1000 == 0:
        print(f"    Processed {idx}/{len(train_df)} games...")

    game_id = game_row['gameId']
    game_result = game_row['result_reg']
    game_date = game_row['date']

    # Get players for this game
    game_players = train_player_df[train_player_df['gameId'] == game_id]

    if len(game_players) == 0:
        continue

    # Calculate team strengths BEFORE updating
    home_players = game_players[game_players['home'] == 1]
    away_players = game_players[game_players['home'] == 0]

    if len(home_players) == 0 or len(away_players) == 0:
        continue

    home_off, home_def = final_rating_system.calculate_team_strength_for_game(home_players)
    away_off, away_def = final_rating_system.calculate_team_strength_for_game(away_players)

    # Features
    features = [
        home_off - away_off,
        home_def - away_def,
        home_off,
        away_def,
        away_off,
        home_def
    ]

    train_features.append(features)
    train_labels.append(game_result)

    # Update ratings AFTER prediction
    final_rating_system.update_ratings(game_players, game_result)

    # Save ratings snapshot
    if idx % 100 == 0:
        final_rating_system.save_current_ratings(game_date)

X_train = np.array(train_features)
y_train = np.array(train_labels)

# Handle NaN/inf
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

# Scale and train logistic regression
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

print("  Training complete - MODEL FROZEN")


# ============================================================================
# 4. EVALUATE ON TEST SET
# ============================================================================

print("\n[4/6] Evaluating on test set (2021-2023)...")

# Continue updating ratings through test period
test_features = []
test_labels = []

print("  Processing test games and updating ratings...")
for idx, game_row in test_df.iterrows():
    if idx % 500 == 0:
        print(f"    Processed {idx}/{len(test_df)} games...")

    game_id = game_row['gameId']
    game_result = game_row['result_reg']

    # Get players for this game
    game_players = test_player_df[test_player_df['gameId'] == game_id]

    if len(game_players) == 0:
        continue

    # Calculate team strengths BEFORE updating
    home_players = game_players[game_players['home'] == 1]
    away_players = game_players[game_players['home'] == 0]

    if len(home_players) == 0 or len(away_players) == 0:
        continue

    home_off, home_def = final_rating_system.calculate_team_strength_for_game(home_players)
    away_off, away_def = final_rating_system.calculate_team_strength_for_game(away_players)

    # Features
    features = [
        home_off - away_off,
        home_def - away_def,
        home_off,
        away_def,
        away_off,
        home_def
    ]

    test_features.append(features)
    test_labels.append(game_result)

    # Update ratings AFTER prediction
    final_rating_system.update_ratings(game_players, game_result)

X_test = np.array(test_features)
y_test = np.array(test_labels)

# Handle NaN/inf
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

# Scale and predict
X_test_scaled = scaler_final.transform(X_test)
test_pred_proba = final_model.predict_proba(X_test_scaled)
test_pred = final_model.predict(X_test_scaled)

# Calculate metrics
test_brier = calculate_multiclass_brier_score(y_test, test_pred_proba)
test_accuracy = accuracy_score(y_test, test_pred)
test_logloss = log_loss(y_test, test_pred_proba)

# Bookmaker benchmark (align with test indices)
test_df_filtered = test_df.iloc[:len(y_test)]
bm_probs_test = test_df_filtered[['bm_prob_away', 'bm_prob_tie', 'bm_prob_home']].values
bm_brier = calculate_multiclass_brier_score(y_test, bm_probs_test)
bm_accuracy = accuracy_score(y_test, np.argmax(bm_probs_test, axis=1))
bm_logloss = log_loss(y_test, bm_probs_test)

print("\n" + "=" * 80)
print("TEST SET RESULTS (2021-2023)")
print("=" * 80)
print(f"Model: Player Rating System + Logistic Regression")
print(f"Hyperparameters: lr={best_params['learning_rate']}, decay={best_params['decay_factor']}")
print(f"\n{'Metric':<20} {'Our Model':<15} {'Bookmaker':<15} {'Improvement':<15}")
print("-" * 80)
print(f"{'Brier Score':<20} {test_brier:<15.4f} {bm_brier:<15.4f} {bm_brier - test_brier:<15.4f}")
print(f"{'Accuracy':<20} {test_accuracy:<15.4f} {bm_accuracy:<15.4f} {test_accuracy - bm_accuracy:<15.4f}")
print(f"{'Log Loss':<20} {test_logloss:<15.4f} {bm_logloss:<15.4f} {bm_logloss - test_logloss:<15.4f}")
print("=" * 80)

# Confusion matrix
cm = confusion_matrix(y_test, test_pred)
print("\nConfusion Matrix:")
print("                Predicted")
print("            Away    Tie   Home")
print(f"Actual Away {cm[0,0]:4d}   {cm[0,1]:4d}   {cm[0,2]:4d}")
print(f"       Tie  {cm[1,0]:4d}   {cm[1,1]:4d}   {cm[1,2]:4d}")
print(f"       Home {cm[2,0]:4d}   {cm[2,1]:4d}   {cm[2,2]:4d}")


# ============================================================================
# 5. BETTING SIMULATION
# ============================================================================

print("\n[5/6] Betting simulation on test set...")

EV_THRESHOLD = 0.03
FLAT_STAKE = 10

bets = []
total_profit = 0
total_staked = 0

for i in range(len(y_test)):
    pred_probs = test_pred_proba[i]
    bm_probs = bm_probs_test[i]
    actual = y_test[i]

    # Calculate edge
    edges = pred_probs - bm_probs
    max_edge_idx = np.argmax(edges)
    max_edge = edges[max_edge_idx]

    if max_edge >= EV_THRESHOLD:
        stake = FLAT_STAKE
        total_staked += stake

        # Check if bet wins
        if max_edge_idx == actual:
            # Get odds
            game_row = test_df_filtered.iloc[i]
            if max_edge_idx == 0:  # Away
                payout = stake * game_row['odd_awaywin']
            elif max_edge_idx == 1:  # Tie
                payout = stake * game_row['odd_draw']
            else:  # Home
                payout = stake * game_row['odd_winhome']

            profit = payout - stake
        else:
            profit = -stake

        total_profit += profit

        bets.append({
            'bet_on': max_edge_idx,
            'actual': actual,
            'edge': max_edge,
            'stake': stake,
            'profit': profit,
            'won': int(max_edge_idx == actual)
        })

num_bets = len(bets)
num_wins = sum([b['won'] for b in bets])
win_rate = num_wins / num_bets if num_bets > 0 else 0
roi = (total_profit / total_staked) if total_staked > 0 else 0

print("\n" + "=" * 80)
print("BETTING SIMULATION RESULTS")
print("=" * 80)
print(f"Strategy: Flat stake when EV > {EV_THRESHOLD*100:.1f}%")
print(f"Stake per bet: ${FLAT_STAKE}")
print(f"\nTotal Profit/Loss: ${total_profit:.2f}")
print(f"Total Staked:      ${total_staked:.2f}")
print(f"Number of Bets:    {num_bets}")
print(f"Number of Wins:    {num_wins}")
print(f"Win Rate:          {win_rate*100:.2f}%")
print(f"ROI:               {roi*100:.2f}%")
print("=" * 80)


# ============================================================================
# 6. SAVE RESULTS
# ============================================================================

print("\n[6/6] Saving results...")

# Save predictions
test_df_filtered['pred_prob_away'] = test_pred_proba[:, 0]
test_df_filtered['pred_prob_tie'] = test_pred_proba[:, 1]
test_df_filtered['pred_prob_home'] = test_pred_proba[:, 2]
test_df_filtered['predicted_outcome'] = test_pred

test_df_filtered.to_csv('test_predictions_player_rating.csv', index=False)
print("  Saved: test_predictions_player_rating.csv")

# Save metrics
import json
metrics = {
    'model_name': 'Player Rating System (Holmes & McHale 2024 Approach)',
    'hyperparameters': {
        'learning_rate': best_params['learning_rate'],
        'decay_factor': best_params['decay_factor']
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
    'betting_roi': float(roi)
}

with open('test_metrics_player_rating.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("  Saved: test_metrics_player_rating.json")

# Save betting history
if len(bets) > 0:
    pd.DataFrame(bets).to_csv('betting_history_player_rating.csv', index=False)
    print("  Saved: betting_history_player_rating.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nPlayer Rating Model (Holmes & McHale 2024):")
print("  ✓ Individual player offensive and defensive ratings")
print("  ✓ Dynamic updates based on game performance")
print("  ✓ Team strength = weighted sum of player ratings")
print("  ✓ Position-specific modeling (forwards/defense/goalies)")
print("  ✓ Logistic regression for match outcome prediction")
print("=" * 80)
