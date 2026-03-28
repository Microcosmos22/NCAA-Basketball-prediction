from pathlib import Path
from dataloader import *
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm

team_season_stats = pd.read_csv("perteam_perseason_stats.csv")

def match_features_fromIDs(teamID1, teamID2, season, team_season_stats, elo_df=None, massey_df=None, sos_df=None, seed_df=None):
    """
    Compute matchup-level feature vector using non-rolling, season-level stats.
    Includes: Season aggregates, Elo, Massey, SOS, Seed features.
    """
    hist_df = pd.read_csv("seeddiff_to_HistSeedWP.csv")

    hist_seed_wp = dict(
        zip(hist_df["SeedDiff"],
            hist_df["HistSeedWP"])
    )

    # Get last completed season stats for each team
    team_a_stats = team_season_stats.query(
        f"TeamID == {teamID1} and Season == {season-1}"
    ).sort_values("Season").tail(1)

    team_b_stats = team_season_stats.query(
        f"TeamID == {teamID2} and Season == {season-1}"
    ).sort_values("Season").tail(1)

    # Skip if no stats available
    if team_a_stats.empty or team_b_stats.empty:
        return pd.DataFrame()

    seed_diff = (
        team_a_stats["num_seed"].values[0]
        - team_b_stats["num_seed"].values[0]
    )

    # Base season aggregates
    match_features = {
        "SeedDiff": seed_diff,
        "HistSeedWP": hist_seed_wp.get(seed_diff, 0.5),
        "SeedDiff_sq": (team_a_stats["num_seed"].values[0] - team_b_stats["num_seed"].values[0])**2,
        "AbsDiff_Seed": np.abs(team_a_stats["num_seed"].values[0] - team_b_stats["num_seed"].values[0]),
        "NetRtgDiff": float(team_a_stats["NetRtg"].iloc[0] - team_b_stats["NetRtg"].iloc[0]),
        "SeedDiff_x_NetRtg": float(team_a_stats["num_seed"].values[0] - team_b_stats["num_seed"].values[0])*float(team_a_stats["NetRtg"].iloc[0] - team_b_stats["NetRtg"].iloc[0]),

        "PointsForDiff": team_a_stats["PointsFor"].values[0] - team_b_stats["PointsFor"].values[0],
        "PointsAgainstDiff": team_a_stats["PointsAgainst"].values[0] - team_b_stats["PointsAgainst"].values[0],
        "WinRateDiff": team_a_stats["WinRate"].values[0] - team_b_stats["WinRate"].values[0],

        # Shooting efficiency
        "FG_pctDiff": team_a_stats["FG_pct"].values[0] - team_b_stats["FG_pct"].values[0],
        "OppFG_pctDiff": team_a_stats["OppFG_pct"].values[0] - team_b_stats["OppFG_pct"].values[0],

        # Rebounds / turnovers
        "ReboundDiff": team_a_stats["ReboundDiff"].values[0] - team_b_stats["ReboundDiff"].values[0],
        "TurnoverMargin": team_a_stats["TurnoverMargin"].values[0] - team_b_stats["TurnoverMargin"].values[0],

        # Offensive / defensive volume
        "FGA_diff": team_a_stats["FGA"].values[0] - team_b_stats["FGA"].values[0],
        #"FGA3_diff": team_a_stats.get("FGA3", 0).values[0] - team_b_stats.get("FGA3", 0).values[0],
        #"FTA_diff": team_a_stats.get("FTA", 0).values[0] - team_b_stats.get("FTA", 0).values[0],

        # Rebounds and assists
        "OR_diff": team_a_stats["OR"].values[0] - team_b_stats["OR"].values[0],
        "DR_diff": team_a_stats["DR"].values[0] - team_b_stats["DR"].values[0],
        #"Ast_diff": team_a_stats.get("Ast", 0).values[0] - team_b_stats.get("Ast", 0).values[0],

        # Turnovers / fouls
        "TO_diff": team_a_stats["TO"].values[0] - team_b_stats["TO"].values[0],
        #"PF_diff": team_a_stats.get("PF", 0).values[0] - team_b_stats.get("PF", 0).values[0],

        # Home advantage
        #"HomeTeam": 1 if team_a_stats.get("WLoc", "N").values[0] == "H" else 0,
    }

    # Elo difference
    if elo_df is not None:
        elo_a = elo_df.query(f"TeamID == {teamID1} and Season < {season}").sort_values("Season").tail(1)
        elo_b = elo_df.query(f"TeamID == {teamID2} and Season < {season}").sort_values("Season").tail(1)
        match_features["EloDiff"] = (elo_a["Elo"].values[0] if not elo_a.empty else 1500) - \
                                    (elo_b["Elo"].values[0] if not elo_b.empty else 1500)

    # Massey rankings difference
    if massey_df is not None:
        massey_a = massey_df.query(f"TeamID == {teamID1} and Season < {season}").sort_values("Season").tail(1)
        massey_b = massey_df.query(f"TeamID == {teamID2} and Season < {season}").sort_values("Season").tail(1)
        match_features["MasseyDiff"] = (massey_a["OrdinalRank"].values[0] if not massey_a.empty else 999) - \
                                       (massey_b["OrdinalRank"].values[0] if not massey_b.empty else 999)

    # SOS difference
    if sos_df is not None:
        sos_a = sos_df.query(f"TeamID == {teamID1} and Season < {season}").sort_values("Season").tail(1)
        sos_b = sos_df.query(f"TeamID == {teamID2} and Season < {season}").sort_values("Season").tail(1)
        match_features["SOSDiff"] = (sos_a["SOS"].values[0] if not sos_a.empty else 0) - \
                                    (sos_b["SOS"].values[0] if not sos_b.empty else 0)

    # Seed differences
    if seed_df is not None:
        seed_a = seed_df.query(f"TeamID == {teamID1} and Season == {season}")["Seed"].values
        seed_b = seed_df.query(f"TeamID == {teamID2} and Season == {season}")["Seed"].values
        seed_a_val = int(seed_a[0]) if len(seed_a) > 0 else 16
        seed_b_val = int(seed_b[0]) if len(seed_b) > 0 else 16
        match_features["SeedDiff"] = seed_a_val - seed_b_val
        match_features["SeedDiff_sq"] = (seed_a_val - seed_b_val) ** 2

    return pd.DataFrame([match_features])

if __name__ == "__main__":
    import re
    N = 100

    Mgames = pd.concat([MRegulargames, MNCAAgames])[["Season", "DayNum", "WTeamID", "LTeamID"]][:N]
    Mgameswon = Mgames.rename(columns={"WTeamID":"Team1ID", "LTeamID":"Team2ID"})
    Mgameswon["Win"] = 1
    Mgameslost = Mgames.rename(columns={"WTeamID":"Team2ID", "LTeamID":"Team1ID"})
    Mgameslost["Win"] = 0
    Mgamestrain = pd.concat([Mgameswon, Mgameslost])

    print(Mgamestrain.columns)
    print(Mgamestrain.shape)
    print("Total rows:", Mgamestrain.shape[0])
    print("Number of wins (y=1):", Mgamestrain["Win"].sum())
    print("Number of losses (y=0):", (Mgamestrain["Win"] == 0).sum())

    X_list = []
    y_list = []


    for i in range(len(Mgamestrain)):
        row = Mgamestrain.iloc[i]

        features = match_features_fromIDs(
            row["Team1ID"],
            row["Team2ID"],
            row["Season"],
            team_season_stats
        )

        if features.empty:
            continue

        X_list.append(features.iloc[0])  # append row, not DataFrame
        y_list.append(row["Win"])

    X = pd.DataFrame(X_list)#.drop(columns=["Seasontype"])
    X.columns = [
        re.sub(r'[^A-Za-z0-9_]+', '_', col)
        for col in X.columns
    ]
    y = pd.Series(y_list)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import log_loss

    # Features and target


    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Define parameters
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "seed": 42,
        "verbose": 1
    }

    # Train
    model = lgb.train(
        params,
        train_data,
        valid_sets=val_data,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=5)
        ]
    )

    # Make predictions
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    loss = log_loss(y_val, y_pred)
    print("Validation log loss:", loss)


    print(Mgames.columns)
    print(Mgames.shape)
