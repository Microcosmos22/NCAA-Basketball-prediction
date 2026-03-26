from pathlib import Path
from dataloader import *
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm

team_season_stats = pd.read_csv("../perteam_perseason_stats.csv")

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

    y = Mgamestrain["Win"]
    X = []
    for i in range(len(Mgamestrain)):

        features = match_features_fromIDs(Mgamestrain.iloc[i]["Team1ID"], Mgamestrain.iloc[i]["Team2ID"], Mgamestrain.iloc[i]["Season"])
        if features.empty:
            continue
        X.append(features)
    X = pd.concat(X, ignore_index=True)  # combine all single-row DataFrames

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
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        early_stopping_rounds=50,
        verbose_eval=1
    )

    # Make predictions
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    loss = log_loss(y_val, y_pred)
    print("Validation log loss:", loss)


    print(Mgames.columns)
    print(Mgames.shape)
