"""
LaLiga Predictor
----------------
Downloads match data for the last 19 seasons, computes team standings and rolling form,
trains a Machine Learning model, and provides a function to predict future fixtures.
"""

import os
import requests
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from bs4 import BeautifulSoup

# -----------------------------------------------------------------------------
# 1) CONFIGURATION
# -----------------------------------------------------------------------------
# Seasons to process: 2005-2006 up to 2024-25
SEASONS = [
    ("05-06", "2005-08-01", "2006-05-31"),
    ("06-07", "2006-08-01", "2007-05-31"),
    ("07-08", "2007-08-01", "2008-05-31"),
    ("08-09", "2008-08-01", "2009-05-31"),
    ("09-10", "2009-08-01", "2010-05-31"),
    ("10-11", "2010-08-01", "2011-05-31"),
    ("11-12", "2011-08-01", "2012-05-31"),
    ("12-13", "2012-08-01", "2013-05-31"),
    ("13-14", "2013-08-01", "2014-05-31"),
    ("14-15", "2014-08-01", "2015-05-31"),
    ("15-16", "2015-08-01", "2016-05-31"),
    ("16-17", "2016-08-01", "2017-05-31"),
    ("17-18", "2017-08-01", "2018-05-31"),
    ("18-19", "2018-09-01", "2019-05-31"),
    ("19-20", "2019-09-01", "2020-05-31"),
    ("20-21", "2020-09-01", "2021-05-31"),
    ("21-22", "2021-09-01", "2022-05-31"),
    ("22-23", "2022-09-01", "2023-05-31"),
    ("23-24", "2023-08-01", "2024-05-31"),
    ("24-25", "2024-08-01", "2025-05-31"),
]

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/SP1.csv"
DATA_DIR = "data/"

# Create data directory if missing
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 2) DOWNLOAD & COMBINE MATCH DATA
# -----------------------------------------------------------------------------
all_matches = []

for code, start_date, end_date in SEASONS:
    url = BASE_URL.format(season=code)
    local_path = os.path.join(DATA_DIR, f"SP1_{code}.csv")
    print(f"Downloading season {code} data...")
    
    # Download CSV if not already present
    if not os.path.exists(local_path):
        r = requests.get(url)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
    
    # Read and keep only necessary columns
    df = pd.read_csv(local_path, usecols=["Date", "HomeTeam", "AwayTeam",
                                          "FTHG", "FTAG", "FTR", "HS", "AS",
                                            "HST", "AST", "HC", "AC", "HF",
                                              "AF", "HR", "AR", "HY", "AY",
                                                "B365H", "B365D", "B365A"])
    # Parse dates
    df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True,
    errors="coerce"       # any unparsable dates become NaT instead of crashing
    )
    df = df.dropna(subset=["Date"])  # drop rows if date parsing failed
    
    all_matches.append(df)

# Combine all seasons into one DataFrame
matches = pd.concat(all_matches, ignore_index=True)
matches = matches.sort_values("Date").reset_index(drop=True)

# -----------------------------------------------------------------------------
# 3) COMPUTE SEASONAL STANDINGS
# -----------------------------------------------------------------------------
def compute_standings(season_df):
    "Given matches for one season, return standings DataFrame."
    teams = pd.DataFrame({
        "Team": pd.concat([season_df["HomeTeam"], season_df["AwayTeam"]]).unique() # type: ignore
    })
    # initialize stats
    stats = {team: {"W":0, "D":0, "L":0, "GF":0, "GA":0, "SHOTS":0,
                     "SHOTS ON TARGET":0, "CORNERS":0, "FOULS":0,
                     "RED CARDS":0, "YELLOW CARDS":0} for team in teams["Team"]}
    
    # accumulate
    for _, row in season_df.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        hg, ag = int(row["FTHG"]), int(row["FTAG"])
        stats[h]["GF"] += hg; stats[h]["GA"] += ag
        stats[a]["GF"] += ag; stats[a]["GA"] += hg
        stats[h]["SHOTS"] += int(row["HS"])
        stats[a]["SHOTS"] += int(row["AS"])
        stats[h]["SHOTS ON TARGET"] += int(row["HST"])
        stats[a]["SHOTS ON TARGET"] += int(row["AST"])
        stats[h]["CORNERS"] += int(row["HC"])
        stats[a]["CORNERS"] += int(row["AC"])
        stats[h]["FOULS"] += int(row["HF"])
        stats[a]["FOULS"] += int(row["AF"])
        stats[h]["RED CARDS"] += int(row["HR"])
        stats[a]["RED CARDS"] += int(row["AR"])
        stats[h]["YELLOW CARDS"] += int(row["HY"])
        stats[a]["YELLOW CARDS"] += int(row["AY"])
        
        if row["FTR"] == "H":
            stats[h]["W"] += 1; stats[a]["L"] += 1
        elif row["FTR"] == "A":
            stats[a]["W"] += 1; stats[h]["L"] += 1
        else:
            stats[h]["D"] += 1; stats[a]["D"] += 1
    
    # build DataFrame
    records = []
    for team, s in stats.items():
        pts = 3*s["W"] + 1*s["D"]
        gd = s["GF"] - s["GA"]
        records.append({
            "Team": team, "W": s["W"], "D": s["D"], "L": s["L"],
            "GF": s["GF"], "GA": s["GA"], "GD": gd, "Pts": pts,
            "Shots": s["SHOTS"], "Shots on Target": s["SHOTS ON TARGET"],
            "Corners": s["CORNERS"], "Fouls": s["FOULS"],
            "Red Cards": s["RED CARDS"], "Yellow Cards": s["YELLOW CARDS"]
        })
    df = pd.DataFrame(records)
    df = df.sort_values(["Pts", "GD", "GF"], ascending=False).reset_index(drop=True)
    df.index += 1  # 1-based ranking
    df.index.name = "Position"
    return df

# Compute and save standings per season
for code, start_date, end_date in SEASONS:
    season_mask = (matches["Date"] >= start_date) & (matches["Date"] <= end_date)
    standings = compute_standings(matches[season_mask])
    file_path = os.path.join(DATA_DIR, f"standings_{code}.csv")
    standings.to_csv(file_path)
    print(f"Standings for {code} saved.")

# -----------------------------------------------------------------------------
# 4) FEATURE ENGINEERING: ROLLING FORM & GOAL AVERAGES
# -----------------------------------------------------------------------------
# Will compute, for each match, for both home and away team:
# - avg goals scored in last N games
# - avg goals conceded in last N games
# - win rate in last N games
# - shots and shots on target in last N games
# - corners, fouls, and cards in last N games

ROLL_WINDOW = 10

#Find Head-to-Head stats
def compute_head_to_head_stats(hist, home_team, away_team, date, window = ROLL_WINDOW):
    h2h = hist[
        (((hist['HomeTeam'] == home_team) & (hist['AwayTeam'] == away_team)) |
        ((hist['HomeTeam'] == away_team) & (hist['AwayTeam'] == home_team))) &
        (hist['Date'] < date)
    ].sort_values("Date", ascending=False).head(window)

    if h2h.empty:
        return 0.33, 0.33, 0.33

    home_wins = 0
    draws = 0
    away_wins = 0
    for _, row in h2h.iterrows():
        if row['HomeTeam'] == home_team and row['FTR'] == 'H':
            home_wins += 1
        elif row['AwayTeam'] == home_team and row['FTR'] == 'A':
            home_wins += 1
        elif row['FTR'] == 'D':
            draws += 1
        else:
            away_wins += 1
    total = home_wins + draws + away_wins
    return home_wins / total, draws / total, away_wins / total

# Prepare to store features
feat_list = []

# For speed, build a dict of each team's past matches
team_hist = {team: pd.DataFrame(columns=matches.columns) for team in
             pd.concat([matches["HomeTeam"], matches["AwayTeam"]]).unique()}

# Iterate through matches in chronological order
for idx, row in matches.iterrows():
    date = row["Date"]
    h, a = row["HomeTeam"], row["AwayTeam"]
    hg, ag = row["FTHG"], row["FTAG"]
    result = row["FTR"]
    HS, AS = row["HS"], row["AS"]
    hst, ast = row["HST"], row["AST"]
    hc, ac = row["HC"], row["AC"]
    hf, af = row["HF"], row["AF"]
    hr, ar = row["HR"], row["AR"]
    hy, ay = row["HY"], row["AY"]

    def compute_feats(team, is_home):
        """Compute rolling stats for `team` up to, but not including, this match."""
        hist = team_hist[team]
        last_n = hist.tail(ROLL_WINDOW)
        weights = np.linspace(1, 2, len(last_n))  #give more weight to recent matches
        weights /= weights.sum()  #normalize

        # if no history, default to league average (approx 1 goal/game)
        if last_n.empty:
            return {"avg_scored": 1.0, "avg_conceded": 1.0, "win_rate": 0.33,
                    "avg_shots": 10.6, "avg_shots_on_target": 3.5,
                    "avg_corners": 4.8, "avg_fouls": 12.2,
                    "avg_red_cards": 0.13, "avg_yellow_cards": 2.2}
        # compute: depending on home/away, choose FTHG/FTAG
        if is_home:
            scored = last_n["FTHG"]; conceded = last_n["FTAG"]
            wins = (last_n["FTR"] == "H").sum()
            shots = last_n["HS"]
            shots_on_target = last_n["HST"]
            corners = last_n["HC"]
            fouls = last_n["HF"]
            red_cards = last_n["HR"]
            yellow_cards = last_n["HY"]
        else:
            scored = last_n["FTAG"]; conceded = last_n["FTHG"]
            wins = (last_n["FTR"] == "A").sum()
            shots = last_n["AS"]
            shots_on_target = last_n["AST"]
            corners = last_n["AC"]
            fouls = last_n["AF"]
            red_cards = last_n["AR"]
            yellow_cards = last_n["AY"]
        draws = (last_n["FTR"] == "D").sum()
        total = len(last_n)
        return {
            "avg_scored": np.average(scored, weights = weights),
            "avg_conceded": np.average(conceded, weights = weights),
            "win_rate": (wins + (0.5 * draws)) / total,
            "avg_shots": np.average(shots, weights = weights),
            "avg_shots_on_target": np.average(shots_on_target, weights = weights),
            "avg_corners": np.average(corners, weights = weights),
            "avg_fouls": np.average(fouls, weights = weights),
            "avg_red_cards": np.average(red_cards, weights = weights),
            "avg_yellow_cards": np.average(yellow_cards, weights = weights)
        }
    
    # compute features for home & away
    fh = compute_feats(h, is_home=True)
    fa = compute_feats(a, is_home=False)

    # Compute head-to-head stats
    h2h_win, h2h_draw, h2h_loss = compute_head_to_head_stats(matches, h, a, date)

    #fractional odds like 8.0 means 8/1; implied P = 1 / (odds + 1)
    p_h = 1.0 / (row["B365H"] + 1)
    p_d = 1.0 / (row["B365D"] + 1)
    p_a = 1.0 / (row["B365A"] + 1)
    p_sum = p_h + p_d + p_a
    odds_ratio_home = p_h / p_sum
    odds_ratio_draw = p_d / p_sum
    odds_ratio_away = p_a / p_sum

    feat = {
        "idx": idx,
        "HomeTeam": h, "AwayTeam": a,
        "home_avg_scored": fh["avg_scored"],
        "home_avg_conceded": fh["avg_conceded"],
        "home_win_rate": fh["win_rate"],
        "home_avg_shots": fh["avg_shots"],
        "home_avg_shots_on_target": fh["avg_shots_on_target"],
        "home_avg_corners": fh["avg_corners"],
        "home_avg_fouls": fh["avg_fouls"],
        "home_avg_red_cards": fh["avg_red_cards"],
        "home_avg_yellow_cards": fh["avg_yellow_cards"],
        "away_avg_scored": fa["avg_scored"],
        "away_avg_conceded": fa["avg_conceded"],
        "away_win_rate": fa["win_rate"],
        "away_avg_shots": fa["avg_shots"],
        "away_avg_shots_on_target": fa["avg_shots_on_target"],
        "away_avg_corners": fa["avg_corners"],
        "away_avg_fouls": fa["avg_fouls"],
        "away_avg_red_cards": fa["avg_red_cards"],
        "away_avg_yellow_cards": fa["avg_yellow_cards"],
        "Date": date,
        "FTR": result,
        "h2h_win": h2h_win,
        "h2h_draw": h2h_draw,
        "h2h_loss": h2h_loss,
        "odds_ratio_home": odds_ratio_home,
        "odds_ratio_draw": odds_ratio_draw,
        "odds_ratio_away": odds_ratio_away,
    }
    feat_list.append(feat)
    
    # now add this match to both teams' history
    team_hist[h] = pd.concat([team_hist[h], row.to_frame().T], ignore_index=True)
    team_hist[a] = pd.concat([team_hist[a], row.to_frame().T], ignore_index=True)

features = pd.DataFrame(feat_list).set_index("idx")

# Encode target: H -> 0, D -> 1, A -> 2
features["target"] = features["FTR"].map({"H":0, "D":1, "A":2})

# -----------------------------------------------------------------------------
# 5) MODEL TRAINING & ROLLING SEASON-BY-SEASON EVALUATION
# -----------------------------------------------------------------------------
# Will train a fresh model for each split:
#   • Train on seasons 1…n
#   • Test on season n+1
# This simulates predicting a coming season using only data available before it.

base_features = [
    "home_avg_scored", "home_avg_conceded", "home_win_rate",
    "home_avg_shots", "home_avg_shots_on_target", "home_avg_corners",
    "home_avg_fouls", "home_avg_red_cards", "home_avg_yellow_cards",
    "away_avg_scored", "away_avg_conceded", "away_win_rate",
    "away_avg_shots", "away_avg_shots_on_target", "away_avg_corners",
    "away_avg_fouls", "away_avg_red_cards", "away_avg_yellow_cards",
    "h2h_win", "h2h_draw", "h2h_loss",
    "odds_ratio_home", "odds_ratio_draw", "odds_ratio_away"
]

#betting columns only included if available
betting_cols = ["B365H", "B365D", "B365A"]
available_cols = [col for col in betting_cols if col in features.columns]
X = features[base_features + available_cols]

y = features["target"]

# List to collect each split’s accuracy
rolling_results = []

# Loop over splits: for i = 1 … len(SEASONS)-1
for i in range(1, len(SEASONS)):
    test_code, test_start, test_end = SEASONS[i]

    #Get all prior seasons up to but not including the test season 
    train_seasons = [s[0] for s in SEASONS[:i]]

    print(f"\n[Split {i}] Train on seasons: {', '.join(train_seasons)}, test on {test_code}")

    # Build date masks
    train_mask = matches["Date"] < pd.to_datetime(test_start)
    test_mask  = (matches["Date"] >= pd.to_datetime(test_start)) & \
                 (matches["Date"] <= pd.to_datetime(test_end))

    # Slice feature matrix X and target y accordingly
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test,  y_test  = X.loc[test_mask],  y.loc[test_mask]

    # Initialize and train a model using ONLY pre-match features
    clf = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.1,
        min_child_weight=3,
        eval_metric='mlogloss',
        random_state=42
    )
    
    clf.fit(X_train, y_train)

    # Predict on the test season and compute accuracy
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f" → Accuracy on season {test_code}: {acc:.3f} | F1: {f1:.3f}")

    importances = clf.feature_importances_
    plt.figure(figsize=(10,6))
    plt.barh(X.columns, importances)
    plt.title(f"Feature Importance for Season {test_code}")
    plt.tight_layout()
    #Save to file
    plt.savefig(os.path.join(DATA_DIR, f"importance_{test_code}.png"))

    # Record the result
    rolling_results.append({
        "Season": test_code,
        "TrainedOn": ", ".join(train_seasons),
        "Accuracy": acc,
        "F1": f1
    })

# Convert to DataFrame, save or inspect
rolling_df = pd.DataFrame(rolling_results)
acc_path = os.path.join(DATA_DIR, "rolling_season_accuracy.csv")
rolling_df.to_csv(acc_path, index = False)
print("\n✔ Rolling season-by-season accuracies saved to 'data/rolling_season_accuracy.csv'")

# -----------------------------------------------------------------------------
# 5b) FINAL MODEL TRAINING & SAVE FOR DEPLOYMENT
# -----------------------------------------------------------------------------
# Train on all data available before the most recent season (2024-25)

final_cutoff = pd.to_datetime(SEASONS[-1][1])  # start date of 24-25
final_mask = matches["Date"] < final_cutoff

X_final = X.loc[final_mask]
y_final = y.loc[final_mask]

final_model = XGBClassifier(
    n_estimators=500, 
    max_depth=8,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.1,
    min_child_weight=3,
    eval_metric='mlogloss',
    random_state=42
)

final_model.fit(X_final, y_final)

# Save the single “production” model
model_path = os.path.join(DATA_DIR, "lali_model.pkl")
joblib.dump(final_model, model_path)
print("✔ Final model trained on all past seasons and saved to 'data/lali_model.pkl'")


# -----------------------------------------------------------------------------
# 6) FUTURE FIXTURES PREDICTION FUNCTION
# -----------------------------------------------------------------------------
def predict_future(fixtures_df, model_path=os.path.join(DATA_DIR, "lali_model.pkl")):
    """
    Predicts outcomes for future fixtures.
    
    fixtures_df must contain columns: Date (datetime), HomeTeam, AwayTeam.
    Returns a DataFrame with probabilities and predicted outcome.
    """
    # Load trained model
    clf = joblib.load(model_path)
    
    # Ensure sorting
    fixtures_df = fixtures_df.copy()
    fixtures_df["Date"] = pd.to_datetime (
    fixtures_df["Date"],
    dayfirst=True,
    errors="coerce"       # any unparsable dates become NaT instead of crashing
    )
    fixtures_df = fixtures_df.sort_values("Date").reset_index(drop=True)

    # We need access to all past matches up to each fixture date:
    hist = matches.copy()
    
    rows = []
    for _, row in fixtures_df.iterrows():
        date, h, a = row["Date"], row["HomeTeam"], row["AwayTeam"]
        
        # compute rolling stats direcly from hist
        def cf(team, is_home):
            #Grab last ROLL_WINDOW matches before this date
            last = (
                hist[
                    ((hist.HomeTeam == team) | (hist.AwayTeam == team)) &
                    (hist.Date < date)
                ]
            )
            if last.empty:
                # Return 9 default values matching the expected features
                return 1.0, 1.0, 0.33, 10.6, 3.5, 4.8, 12.2, 0.13, 2.2
            if is_home:
                s = last["FTHG"]; c = last["FTAG"]
                wins = (last["FTR"] == "H").sum()
                shots = last["HS"]
                shots_on_target = last["HST"]
                corners = last["HC"]
                fouls = last["HF"]
                red_cards = last["HR"]
                yellow_cards = last["HY"]
            else:
                s = last["FTAG"]; c = last["FTHG"]
                wins = (last["FTR"] == "A").sum()
                shots = last["AS"]
                shots_on_target = last["AST"]
                corners = last["AC"]
                fouls = last["AF"]
                red_cards = last["AR"]
                yellow_cards = last["AY"]
            draws = (last["FTR"] == "D").sum()
            total = len(last)
            return (
                s.mean(), c.mean(), (wins + 0.5*draws)/total,
                shots.mean(), shots_on_target.mean(), corners.mean(),
                fouls.mean(), red_cards.mean(), yellow_cards.mean()
            )

        hs, hc, hw, hsh, hsht, hco, hf, hr, hy = cf(h, True)
        as_, ac, aw, ash, asht, aco, af, ar, ay = cf(a, False)

        h2h_win, h2h_draw, h2h_loss = compute_head_to_head_stats(hist, h, a, date) 

        if all(c in row and pd.notna(row[c]) for c in ("B365H","B365D","B365A")):
            p_h = 1.0/(row["B365H"]+1)
            p_d = 1.0/(row["B365D"]+1)
            p_a = 1.0/(row["B365A"]+1)
            p_sum = p_h + p_d + p_a
            or_home = p_h / p_sum
            or_draw = p_d / p_sum
            or_away = p_a / p_sum
        else:
            or_home, or_draw, or_away = 0.33,0.33,0.33


        #Compute base features
        values = [
            hs, hc, hw, hsh, hsht, hco, hf, hr, hy,
            as_, ac, aw, ash, asht, aco, af, ar, ay,
            h2h_win, h2h_draw, h2h_loss,
            or_home, or_draw, or_away
        ]
        cols = base_features.copy()

        Xf_df = pd.DataFrame([values], columns = cols)

        probs = clf.predict_proba(Xf_df)[0]
        pred = clf.predict(Xf_df)[0]

        mapping = {0:"Home Win",1:"Draw",2:"Away Win"}
        
        rows.append({
            "Date": date,
            "HomeTeam": h, "AwayTeam": a,
            "P(Home Win)": probs[0],
            "P(Draw)": probs[1],
            "P(Away Win)": probs[2],
            "Prediction": mapping[pred]
        })
    
    return pd.DataFrame(rows)

# Example usage:
future = pd.DataFrame([
    {
        "Date": "16/08/2025",
        "HomeTeam": "Barcelona",
        "AwayTeam": "Real Madrid",
        "B365H": 0.9090,
        "B365D": 3.34,
        "B365A": 2.56
    },
    {
        "Date": "17/08/2025",
        "HomeTeam": "Ath Madrid",
        "AwayTeam": "Sevilla"
        # Odds missing here
    },
    {
        "Date": "18/08/2025",
        "HomeTeam": "Valladolid",
        "AwayTeam": "Barcelona",
        "B365H": 7.14,
        "B365D": 4.75,
        "B365A": 0.333333
    }
])
preds = predict_future(future)
print(preds)
