"""
============================================
NHL DATA PREPARATION SCRIPT
============================================
This script prepares the NHL game data for prediction modeling.
It combines gamedata.csv and playergamedata.csv into a single dataset.

Requirements:
- gamedata.csv (game-level data with odds)
- playergamedata.csv (player-level data)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("NHL Data Preparation")
print("=" * 60)

# ============================================
# 1. LOAD RAW DATA
# ============================================
print("\n[1/4] Loading raw data files...")

try:
    # Load game-level data
    gamedata = pd.read_csv('gamedata.csv')
    print(f"  Loaded gamedata.csv: {len(gamedata):,} rows")

    # Load player-level data
    playerdata = pd.read_csv('playergamedata.csv')
    print(f"  Loaded playergamedata.csv: {len(playerdata):,} rows")

except FileNotFoundError as e:
    print(f"\nERROR: {e}")
    print("\nPlease ensure both data files are in the current directory:")
    print("  - gamedata.csv")
    print("  - playergamedata.csv")
    exit(1)

# ============================================
# 2. PREPARE GAME-LEVEL DATA
# ============================================
print("\n[2/4] Preparing game-level data...")

# Convert date to datetime
gamedata['date'] = pd.to_datetime(gamedata['date'])

# Sort by date to ensure chronological order
gamedata = gamedata.sort_values('date').reset_index(drop=True)

# Check for missing odds
missing_odds = gamedata[['odd_winhome', 'odd_draw', 'odd_awaywin']].isnull().any(axis=1).sum()
print(f"  Games with missing odds: {missing_odds}")

# Remove games with missing odds (can't bet without odds)
gamedata = gamedata.dropna(subset=['odd_winhome', 'odd_draw', 'odd_awaywin'])
print(f"  Games after removing missing odds: {len(gamedata):,}")

# Verify result_reg values
result_counts = gamedata['result_reg'].value_counts().sort_index()
print(f"\n  Result distribution:")
print(f"    Away wins (0): {result_counts.get(0, 0):,}")
print(f"    Ties (1): {result_counts.get(1, 0):,}")
print(f"    Home wins (2): {result_counts.get(2, 0):,}")

# ============================================
# 3. AGGREGATE PLAYER DATA (OPTIONAL)
# ============================================
print("\n[3/4] Aggregating player statistics...")

# Calculate team-level statistics from player data
# Group by gameId and teamname to get team totals
team_stats = playerdata.groupby(['gameId', 'teamname', 'home']).agg({
    'goals': 'sum',
    'assists': 'sum',
    'shots': 'sum',
    'toi_seconds': 'sum',
    'playerId': 'count'  # Number of players
}).reset_index()

team_stats.columns = ['gameId', 'teamname', 'home', 'team_goals_from_players',
                      'team_assists', 'team_shots', 'team_toi_total', 'num_players']

# Separate home and away team stats
home_stats = team_stats[team_stats['home'] == 1].copy()
away_stats = team_stats[team_stats['home'] == 0].copy()

# Rename columns for home team
home_stats = home_stats.rename(columns={
    'teamname': 'teamname_home',
    'team_goals_from_players': 'home_goals_from_players',
    'team_assists': 'home_assists',
    'team_shots': 'home_shots',
    'team_toi_total': 'home_toi_total',
    'num_players': 'home_num_players'
})
home_stats = home_stats.drop('home', axis=1)

# Rename columns for away team
away_stats = away_stats.rename(columns={
    'teamname': 'teamname_away',
    'team_goals_from_players': 'away_goals_from_players',
    'team_assists': 'away_assists',
    'team_shots': 'away_shots',
    'team_toi_total': 'away_toi_total',
    'num_players': 'away_num_players'
})
away_stats = away_stats.drop('home', axis=1)

# Merge player stats with game data
data = gamedata.merge(home_stats, on='gameId', how='left', suffixes=('', '_home_dup'))
data = data.merge(away_stats, on='gameId', how='left', suffixes=('', '_away_dup'))

# Verify the merge
print(f"  Combined dataset: {len(data):,} games")

# ============================================
# 4. SAVE PROCESSED DATA
# ============================================
print("\n[4/4] Saving processed data...")

# Save the combined dataset
output_file = 'se_assignment1_1_data.csv'
data.to_csv(output_file, index=False)
print(f"  Saved to: {output_file}")

# Display data summary
print("\n" + "=" * 60)
print("DATA SUMMARY")
print("=" * 60)
print(f"Total games: {len(data):,}")
print(f"Date range: {data['date'].min()} to {data['date'].max()}")
print(f"Number of columns: {len(data.columns)}")
print(f"\nSeasons covered:")
season_counts = data['season'].value_counts().sort_index()
for season, count in season_counts.items():
    print(f"  {season}: {count:,} games")

print("\n" + "=" * 60)
print("Data preparation complete!")
print("=" * 60)
print(f"\nNext step: Run the prediction model using {output_file}")
