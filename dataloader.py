from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm

DATA = Path("./data")

CFG = {'train_path': DATA / "train.csv"

}

"""
Each training row corresponds to a single matchup, but most predictive power comes from season-level aggregated features like win rate, point differential, shooting efficiency, etc.

So the workflow is:

1. Compute global/team-season stats from all games in that season.
2. Build match-level rows by combining Team A and Team B features (or differences).
3. Train the model on these match-level rows with the target being “did Team A win?”.

Without the aggregated stats, each row would only have the raw box scores of a single game, which isn’t enough for accurate tournament predictions.
"""

def compute_rolling_seasonal_stats(twoteam_level):
    g = twoteam_level.groupby(
        ["TeamID", "Season", "Seasontype"]
    )
    team_season_stats = twoteam_level.copy()

    team_season_stats["GamesPlayed"] = g.cumcount()

    team_season_stats["WinsSoFar"] = (
        g["Win"].cumsum()
        - team_season_stats["Win"]
    )

    team_season_stats["PointsForSoFar"] = (
        g["PointsFor"].cumsum()
        - team_season_stats["PointsFor"]
    )

    team_season_stats["PointsAgainstSoFar"] = (
        g["PointsAgainst"].cumsum()
        - team_season_stats["PointsAgainst"]
    )

    team_season_stats["AvgPointsFor"] = (
        team_season_stats["PointsForSoFar"]
        / team_season_stats["GamesPlayed"]
    )

    team_season_stats["AvgPointsAgainst"] = (
        team_season_stats["PointsAgainstSoFar"]
        / team_season_stats["GamesPlayed"]
    )

    team_season_stats["WinRate"] = (
        team_season_stats["WinsSoFar"]
        / team_season_stats["GamesPlayed"]
    )

    team_season_stats = team_season_stats.replace(
        [np.inf, -np.inf],
        np.nan
    )

    team_season_stats["GamesPlayed"] = g.cumcount()

    team_season_stats["FGM_sofar"] = g["FGM"].cumsum() - team_season_stats["FGM"]
    team_season_stats["FGA_sofar"] = g["FGA"].cumsum() - team_season_stats["FGA"]

    team_season_stats["OppFGM_sofar"] = g["OppFGM"].cumsum() - team_season_stats["OppFGM"]
    team_season_stats["OppFGA_sofar"] = g["OppFGA"].cumsum() - team_season_stats["OppFGA"]

    team_season_stats["OR_sofar"] = g["OR"].cumsum() - team_season_stats["OR"]
    team_season_stats["DR_sofar"] = g["DR"].cumsum() - team_season_stats["DR"]

    team_season_stats["OppOR_sofar"] = g["OppOR"].cumsum() - team_season_stats["OppOR"]
    team_season_stats["OppDR_sofar"] = g["OppDR"].cumsum() - team_season_stats["OppDR"]

    team_season_stats["TO_sofar"] = g["TO"].cumsum() - team_season_stats["TO"]
    team_season_stats["OppTO_sofar"] = g["OppTO"].cumsum() - team_season_stats["OppTO"]

    team_season_stats["FG_pct"] = team_season_stats["FGM_sofar"] / team_season_stats["FGA_sofar"]

    team_season_stats["OppFG_pct"] = (
        team_season_stats["OppFGM_sofar"] /
        team_season_stats["OppFGA_sofar"]
    )

    team_season_stats["ReboundDiff"] = (
        (team_season_stats["OR_sofar"] + team_season_stats["DR_sofar"])
        -
        (team_season_stats["OppOR_sofar"] + team_season_stats["OppDR_sofar"])
    )

    team_season_stats["TurnoverMargin"] = (
        team_season_stats["OppTO_sofar"]
        -
        team_season_stats["TO_sofar"]
    )
    return team_season_stats

def create_winrows_loserows(Regulargames):
    winrows = pd.DataFrame({})
    winrows = Regulargames[[
        "Season", "DayNum",
        "WTeamID", "LTeamID",
        "WScore", "LScore",
        "WLoc",

        "WFGM", "WFGA", "WFGM3", "WFGA3",
        "WFTM", "WFTA", "WOR", "WDR",
        "WAst", "WTO", "WStl", "WBlk", "WPF",

        "LFGM", "LFGA", "LFGM3", "LFGA3",
        "LFTM", "LFTA", "LOR", "LDR",
        "LAst", "LTO", "LStl", "LBlk", "LPF"
    ]]

    winrows = winrows.rename(columns={

        # IDs
        "WTeamID": "TeamID",
        "LTeamID": "OpponentID",

        # scores
        "WScore": "PointsFor",
        "LScore": "PointsAgainst",

        # team stats
        "WFGM": "FGM",
        "WFGA": "FGA",
        "WFGM3": "FGM3",
        "WFGA3": "FGA3",
        "WFTM": "FTM",
        "WFTA": "FTA",
        "WOR": "OR",
        "WDR": "DR",
        "WAst": "Ast",
        "WTO": "TO",
        "WStl": "Stl",
        "WBlk": "Blk",
        "WPF": "PF",

        # opponent stats
        "LFGM": "OppFGM",
        "LFGA": "OppFGA",
        "LFGM3": "OppFGM3",
        "LFGA3": "OppFGA3",
        "LFTM": "OppFTM",
        "LFTA": "OppFTA",
        "LOR": "OppOR",
        "LDR": "OppDR",
        "LAst": "OppAst",
        "LTO": "OppTO",
        "LStl": "OppStl",
        "LBlk": "OppBlk",
        "LPF": "OppPF"
    })

    winrows["Win"] = 1
    winrows["Seasontype"] = "regular"

    loserows = pd.DataFrame({})
    loserows = Regulargames[[
        "Season", "DayNum",
        "WTeamID", "LTeamID",
        "WScore", "LScore",
        "WLoc",

        "WFGM", "WFGA", "WFGM3", "WFGA3",
        "WFTM", "WFTA", "WOR", "WDR",
        "WAst", "WTO", "WStl", "WBlk", "WPF",

        "LFGM", "LFGA", "LFGM3", "LFGA3",
        "LFTM", "LFTA", "LOR", "LDR",
        "LAst", "LTO", "LStl", "LBlk", "LPF"
    ]]
    loserows = loserows.rename(columns={

        "LTeamID": "TeamID",
        "WTeamID": "OpponentID",

        "LScore": "PointsFor",
        "WScore": "PointsAgainst",

        # team stats
        "LFGM": "FGM",
        "LFGA": "FGA",
        "LFGM3": "FGM3",
        "LFGA3": "FGA3",
        "LFTM": "FTM",
        "LFTA": "FTA",
        "LOR": "OR",
        "LDR": "DR",
        "LAst": "Ast",
        "LTO": "TO",
        "LStl": "Stl",
        "LBlk": "Blk",
        "LPF": "PF",

        # opponent stats
        "WFGM": "OppFGM",
        "WFGA": "OppFGA",
        "WFGM3": "OppFGM3",
        "WFGA3": "OppFGA3",
        "WFTM": "OppFTM",
        "WFTA": "OppFTA",
        "WOR": "OppOR",
        "WDR": "OppDR",
        "WAst": "OppAst",
        "WTO": "OppTO",
        "WStl": "OppStl",
        "WBlk": "OppBlk",
        "WPF": "OppPF"
    })
    loserows["Win"] = 0
    loserows["Seasontype"] = "regular"

    return winrows, loserows

MRegulargames = pd.read_csv("data/MRegularSeasonDetailedResults.csv")
MNCAAgames = pd.read_csv("data/MNCAATourneyCompactResults.csv")
MNCAATourneySeeds = pd.read_csv("data/MNCAATourneySeeds.csv")
MTeams = pd.read_csv("data/MTeams.csv")
MMasseyOrdinals = pd.read_csv("data/MMasseyOrdinals.csv")

if __name__ == "__main__":
    #MRegularSeasonDetailedResults = pd.read_csv("data/MRegularSeasonDetailedResults.csv")
    #MRegularSeasonDetailedResults = pd.read_csv("data/MRegularSeasonDetailedResults.csv")
    #MRegularSeasonDetailedResults = pd.read_csv("data/MRegularSeasonDetailedResults.csv")
    #MRegularSeasonDetailedResults = pd.read_csv("data/MRegularSeasonDetailedResults.csv")
    #MRegularSeasonDetailedResults = pd.read_csv("data/MRegularSeasonDetailedResults.csv")

    """ ################## Double the Regular games so that we can count the
    pd[team][season][seasontype] stats with groupby #################### """

    winrows, loserows = create_winrows_loserows()
    twoteam_level = pd.concat([winrows.copy(), loserows.copy()], ignore_index=True)

    """ ################## Double the NCAA games so that we can count the
    pd[team][season][seasontype] stats with groupby #################### """

    winrows = pd.DataFrame({})
    winrows = MNCAAgames[["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"]]
    winrows = winrows.rename(columns={
        "WTeamID": "TeamID",
        "LTeamID": "OpponentID",
        "WScore": "PointsFor",
        "LScore": "PointsAgainst"
        })
    winrows["Win"] = 1
    winrows["Seasontype"] = "NCAA"

    loserows = pd.DataFrame({})
    loserows = MNCAAgames[["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"]]
    loserows = loserows.rename(columns={
        "LTeamID": "TeamID",
        "WTeamID": "OpponentID",
        "LScore": "PointsFor",
        "WScore": "PointsAgainst"
        })
    loserows["Win"] = 0
    loserows["Seasontype"] = "NCAA"

    twoteam_level = pd.concat([twoteam_level, winrows, loserows], ignore_index=True)
    twoteam_level = twoteam_level.sort_values(["TeamID", "Season", "DayNum"])

    """ ####################################### """

    twoteam_level = twoteam_level.sort_values(
        ["TeamID", "Season", "Seasontype", "DayNum"]
    )

    team_season_stats = compute_rolling_seasonal_stats(twoteam_level)

    print(MRegulargames.shape)
    print(MNCAAgames.shape)

    print(twoteam_level.shape)
    print(twoteam_level.head())

    print(team_season_stats.head())
    print(team_season_stats.columns)

    #team_season_stats.to_csv("perteam_perseason_stats.csv")
    """ ####################################################### """

    #print(team_season_stats)
