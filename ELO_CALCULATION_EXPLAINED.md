# ELO Rating Calculation - Detailed Explanation

## Overview

Our NHL prediction models use **ELO ratings** to measure team strength over time. ELO ratings dynamically update after each game based on the result and the difference in team strength.

This approach follows **Hvattum & Arntzen (2010)** methodology for ice hockey prediction.

---

## ELO Rating System Components

### 1. **Initialization**

All teams start with the same rating:

```
Initial ELO = 1500 points
```

**Code implementation:**
```python
def initialize_elo_ratings(teams, initial_rating=1500):
    """Initialize ELO ratings for all teams."""
    return {team: initial_rating for team in teams}
```

---

### 2. **Key Hyperparameters**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| **K-factor** | 20 | Learning rate - how much ratings change per game |
| **Home Advantage** | 100 points | ELO boost given to home team |

**Why these values?**
- K-factor = 20: Standard for sports, balances responsiveness vs stability
- Home advantage = 100: Represents ~6-8% win probability boost for home teams in NHL

---

## ELO Update Formula

After each game, both teams' ELO ratings are updated using this process:

### **Step 1: Adjust for Home Advantage**

```
Home Rating (Adjusted) = Home Rating + Home Advantage
                       = Home Rating + 100
```

This gives the home team a temporary boost for expected score calculation only.

### **Step 2: Calculate Expected Scores**

Using the standard ELO formula:

```
Expected_Home = 1 / (1 + 10^((Away_Rating - Home_Rating_Adjusted) / 400))

Expected_Away = 1 - Expected_Home
```

**What this means:**
- If teams are equal (both 1500), home team has ~64% expected score due to home advantage
- The 400 scaling factor determines how rating differences translate to win probabilities
- 10^(difference/400) is the standard chess ELO formula

**Example calculation:**
```
Home team: 1600 ELO
Away team: 1500 ELO
Home adjusted: 1600 + 100 = 1700

Expected_Home = 1 / (1 + 10^((1500-1700)/400))
              = 1 / (1 + 10^(-0.5))
              = 1 / (1 + 0.316)
              = 0.760  (76% expected score)

Expected_Away = 1 - 0.760 = 0.240
```

### **Step 3: Determine Actual Scores**

Based on game result:

| Result | Home Actual | Away Actual |
|--------|-------------|-------------|
| **Home Win** (2) | 1.0 | 0.0 |
| **Tie** (1) | 0.5 | 0.5 |
| **Away Win** (0) | 0.0 | 1.0 |

**Code:**
```python
if result == 2:  # Home win
    actual_home, actual_away = 1.0, 0.0
elif result == 1:  # Tie
    actual_home, actual_away = 0.5, 0.5
else:  # Away win
    actual_home, actual_away = 0.0, 1.0
```

### **Step 4: Update Ratings**

```
New_Home_Rating = Old_Home_Rating + K × (Actual_Home - Expected_Home)

New_Away_Rating = Old_Away_Rating + K × (Actual_Away - Expected_Away)
```

**Code:**
```python
new_home_rating = home_rating + k_factor * (actual_home - expected_home)
new_away_rating = away_rating + k_factor * (actual_away - expected_away)
```

---

## Complete Example

**Scenario:**
- Home team (Toronto Maple Leafs): 1600 ELO
- Away team (Montreal Canadiens): 1500 ELO
- Result: Home team wins
- K-factor: 20
- Home advantage: 100

**Step-by-step:**

### 1. Adjust for home advantage:
```
Home adjusted = 1600 + 100 = 1700
```

### 2. Calculate expected scores:
```
Expected_Home = 1 / (1 + 10^((1500-1700)/400))
              = 1 / (1 + 10^(-0.5))
              = 0.760

Expected_Away = 1 - 0.760 = 0.240
```

### 3. Actual scores (home win):
```
Actual_Home = 1.0
Actual_Away = 0.0
```

### 4. Update ratings:
```
New_Home = 1600 + 20 × (1.0 - 0.760)
         = 1600 + 20 × 0.240
         = 1600 + 4.8
         = 1604.8

New_Away = 1500 + 20 × (0.0 - 0.240)
         = 1500 + 20 × (-0.240)
         = 1500 - 4.8
         = 1495.2
```

**Result:**
- Toronto: 1600 → 1604.8 (+4.8)
- Montreal: 1500 → 1495.2 (-4.8)

**Interpretation:** Toronto gained 4.8 points for winning, but less than the full 20 points because they were already favored (76% expected score). Montreal lost 4.8 points.

---

## Key Properties of ELO System

### 1. **Zero-Sum Game**
Points gained by winner = Points lost by loser
```
Toronto +4.8 = Montreal -4.8
```

### 2. **Upset Bonus**
If underdog wins, they gain MORE points:

**Example: Montreal wins instead**
```
Expected_Home = 0.760
Expected_Away = 0.240

Actual_Home = 0.0  (lost)
Actual_Away = 1.0  (won)

New_Home = 1600 + 20 × (0.0 - 0.760) = 1600 - 15.2 = 1584.8  (lost 15.2)
New_Away = 1500 + 20 × (1.0 - 0.240) = 1500 + 15.2 = 1515.2  (gained 15.2!)
```

Montreal gains 15.2 points (vs only 4.8 if Toronto won) because they were the underdog.

### 3. **Ties Give Half-Points**
Both teams move toward equilibrium:
```
Actual_Home = 0.5
Actual_Away = 0.5

New_Home = 1600 + 20 × (0.5 - 0.760) = 1600 - 5.2 = 1594.8
New_Away = 1500 + 20 × (0.5 - 0.240) = 1500 + 5.2 = 1505.2
```

Still zero-sum, but smaller changes than a decisive result.

---

## Implementation in Our Model

### Sequential Calculation (No Data Leakage)

**Critical:** We calculate ELO ratings chronologically and store the rating **BEFORE** each game:

```python
# Sort all games by date
full_df = df.sort_values('date')

# Initialize ratings
elo_ratings = initialize_elo_ratings(all_teams, initial_rating=1500)

# Track ratings BEFORE each game
elo_home_before = []
elo_away_before = []

for idx, row in full_df.iterrows():
    home_team = row['teamname_home']
    away_team = row['teamname_away']
    result = row['result_reg']

    # STORE rating BEFORE game (prevents data leakage)
    elo_home_before.append(elo_ratings[home_team])
    elo_away_before.append(elo_ratings[away_team])

    # UPDATE ratings AFTER storing
    new_home, new_away = update_elo_ratings(
        elo_ratings[home_team],
        elo_ratings[away_team],
        result,
        k_factor=20,
        home_advantage=100
    )

    elo_ratings[home_team] = new_home
    elo_ratings[away_team] = new_away

# Add to dataframe
df['elo_home'] = elo_home_before
df['elo_away'] = elo_away_before
df['elo_diff'] = df['elo_home'] - df['elo_away']
```

**Why this order matters:**
1. We store ratings BEFORE the game
2. These are the ratings available for prediction
3. We update ratings AFTER storing
4. This prevents using information from the current game to predict itself (data leakage)

---

## ELO as a Feature

After calculation, we use **ELO difference** as a primary feature:

```
ELO_diff = Home_ELO - Away_ELO
```

**Example values:**
- ELO_diff = +200 → Home team much stronger (expected ~75% win probability)
- ELO_diff = 0 → Teams equal strength (home advantage still applies)
- ELO_diff = -200 → Away team much stronger

This single feature captures:
- Team quality (historical performance)
- Recent form (recent results affect current ELO)
- Relative strength between opponents

---

## ELO Rating Distribution

After several seasons, team ELO ratings typically range:

| Rating Range | Interpretation | Example |
|--------------|----------------|---------|
| 1650+ | Elite team | Colorado Avalanche 2022 |
| 1550-1650 | Strong team | Most playoff teams |
| 1450-1550 | Average team | Middle of standings |
| 1350-1450 | Weak team | Bottom teams |
| <1350 | Very weak team | Rebuilding teams |

**Mean reversion:** Teams trend toward 1500 over time unless consistently winning/losing.

---

## Comparison to Other Approaches

### ELO vs Fixed Team Ratings
✅ **ELO adapts** to recent performance
❌ Fixed ratings ignore current form

### ELO vs Win Percentage
✅ **ELO accounts for strength of opponent** (beating strong team = more points)
❌ Win % treats all wins equally

### ELO vs Complex Models
✅ **ELO is simple and interpretable**
❌ May miss factors like injuries, roster changes

---

## Hyperparameter Selection

We chose K=20 and Home=100 based on:

1. **Literature:** Hvattum & Arntzen (2010) used similar values for ice hockey
2. **Domain knowledge:** NHL home advantage is well-documented at ~54-56% win rate
3. **Simplicity:** Standard values that don't require tuning

**Could we tune these?**
Yes, but we focused on other aspects:
- K-factor could be tuned (10-30 typical range)
- Home advantage could be tuned (80-120 point range)
- Rolling window size was our main hyperparameter

---

## References

1. **Hvattum, L. M., & Arntzen, H. (2010)**
   - "Using ELO ratings for match result prediction in association football"
   - Original paper establishing ELO for team sports

2. **Elo, A. E. (1978)**
   - "The Rating of Chessplayers, Past and Present"
   - Original ELO system for chess

3. **FiveThirtyEight NHL Ratings**
   - Uses similar ELO-based approach for NHL predictions
   - K ≈ 20, Home advantage varies by team

---

## Complete Code Reference

Full implementation in: `se_assignment1_1_code_simple.py` (lines 100-147, 196-248)

**Key functions:**
- `initialize_elo_ratings()` - Lines 100-102
- `update_elo_ratings()` - Lines 105-147
- Sequential calculation loop - Lines 220-239

---

## Summary

**Our ELO system:**
- ✅ Initializes all teams at 1500
- ✅ Uses K-factor = 20 (learning rate)
- ✅ Applies home advantage = 100 points
- ✅ Updates after each game based on result and expectation
- ✅ Stores ratings BEFORE games to prevent data leakage
- ✅ Creates ELO_diff feature for prediction model

**Result:** Simple, interpretable team strength metric that adapts over time and forms the foundation of our prediction model.
