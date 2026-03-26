from pathlib import Path
from dataloader import *
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm

team_season_stats = pd.read_csv("./perteam_perseason_stats.csv")

def match_features_fromIDs(teamID1, teamID2, season):
    """
    Go to last season, compute average stats for teamA and team B.
    The difference will be the input feature vector
    """
    team_a_stats = team_season_stats.query("TeamID == "+str(teamID1)+" and Season < " + str(season)).sort_values("Season").tail(1)
    team_b_stats = team_season_stats.query("TeamID == "+str(teamID2)+" and Season < " + str(season)).sort_values("Season").tail(1)

    if team_a_stats.empty or team_b_stats.empty:
        return team_a_stats

    match_features = pd.DataFrame([{
        "PointsForDiff": team_a_stats["AvgPointsFor"].values[0] - team_b_stats["AvgPointsFor"].values[0],
        "PointsAgainstDiff": team_a_stats["AvgPointsAgainst"].values[0] - team_b_stats["AvgPointsAgainst"].values[0],
        "WinRateDiff": team_a_stats["WinRate"].values[0] - team_b_stats["WinRate"].values[0],

        # Shooting efficiency
        "FG_pctDiff": team_a_stats["FG_pct"].values[0] - team_b_stats["FG_pct"].values[0],
        "OppFG_pctDiff": team_a_stats["OppFG_pct"].values[0] - team_b_stats["OppFG_pct"].values[0],
        "ReboundDiffDiff": team_a_stats["ReboundDiff"].values[0] - team_b_stats["ReboundDiff"].values[0],
        "TurnoverMarginDiff": team_a_stats["TurnoverMargin"].values[0] - team_b_stats["TurnoverMargin"].values[0],

        # Offensive / defensive volume
        "FGA_diff": team_a_stats["FGA_sofar"].values[0] - team_b_stats["FGA_sofar"].values[0],
        "FGA3_diff": team_a_stats["FGA3"].values[0] - team_b_stats["FGA3"].values[0],
        "FTA_diff": team_a_stats["FTA"].values[0] - team_b_stats["FTA"].values[0],

        # Rebounds and assists
        "OR_diff": team_a_stats["OR_sofar"].values[0] - team_b_stats["OR_sofar"].values[0],
        "DR_diff": team_a_stats["DR_sofar"].values[0] - team_b_stats["DR_sofar"].values[0],
        "Ast_diff": team_a_stats["Ast"].values[0] - team_b_stats["Ast"].values[0],

        # Turnovers / fouls
        "TO_diff": team_a_stats["TO_sofar"].values[0] - team_b_stats["TO_sofar"].values[0],
        "PF_diff": team_a_stats["PF"].values[0] - team_b_stats["PF"].values[0],

        # Home advantage
        "HomeTeam": 1 if team_a_stats["WLoc"].values[0] == "H" else 0
    }])
    return match_features

if __name__ == "__main__":
    import re
    
    Mgames = pd.concat([MRegulargames, MNCAAgames])[["Season", "DayNum", "WTeamID", "LTeamID"]][:1000]
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
            row["Season"]
        )

        if features.empty:
            continue

        X_list.append(features.iloc[0])  # append row, not DataFrame
        y_list.append(row["Win"])

    X = pd.DataFrame(X_list).drop(columns=["Seasontype"])
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
