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

region_map = {
    "W": 0,
    "X": 1,
    "Y": 2,
    "Z": 3
}

def seed_to_features(seed):
    if isinstance(seed, str):
        region = region_map.get(seed[0], -1)
        number = int(seed[1:3])
    else:
        region = -1
        number = 16

    return number

def compute_seasonal_stats(twoteam_level, massey_df=True, elo_df=None, sos_df=None):
    """
    Compute season-level aggregated stats for each team.
    That we access via ["TeamID"]["Season"]

    Also merges optional features:
        - Seed info from twoteam_seed_stats
        - Massey rankings from massey_df
        - Elo ratings from elo_df
        - Strength of schedule from sos_df
    """
    g = twoteam_level.groupby(["TeamID", "Season", "Seasontype"])

    team_season_stats = twoteam_level.copy()
    team_season_stats["GamesPlayed"] = g.cumcount()
    season_aggs = team_season_stats.groupby(["TeamID", "Season"]).agg({
        "Win": "sum",
        "PointsFor": "sum",
        "PointsAgainst": "sum",
        "GamesPlayed": "count",  # or sum if you already have 1 per game
        "FGM": "sum",
        "FGA": "sum",
        "OppFGM": "sum",
        "OppFGA": "sum",
        "OR": "sum",
        "DR": "sum",
        "OppOR": "sum",
        "OppDR": "sum",
        "TO": "sum",
        "OppTO": "sum",

    }).reset_index()
    season_aggs = season_aggs.merge(
        teamid_seed[["Season", "TeamID", "num_seed"]],
        on=["Season", "TeamID"],
        how="left"
    )

    # Win rate
    season_aggs["AvgPointsFor"] = season_aggs["PointsFor"] / season_aggs["GamesPlayed"]
    season_aggs["AvgPointsAgainst"] = season_aggs["PointsAgainst"] / season_aggs["GamesPlayed"]
    season_aggs["NetRtg"] = (season_aggs["AvgPointsFor"] - season_aggs["AvgPointsAgainst"])
    season_aggs["FG_pct"] = season_aggs["FGM"] / season_aggs["FGA"]
    season_aggs["OppFG_pct"] = season_aggs["OppFGM"] / season_aggs["OppFGA"]
    season_aggs["WinRate"] = season_aggs["Win"] / season_aggs["GamesPlayed"]
    season_aggs["ReboundDiff"] = (season_aggs["OR"] + season_aggs["DR"]) - (season_aggs["OppOR"] + season_aggs["OppDR"])
    season_aggs["TurnoverMargin"] = season_aggs["OppTO"] - season_aggs["TO"]
    season_aggs.rename(columns={"Season_x": "Season"}, inplace=True)

    # Merge optional features
    if massey_df is not None:
        systems_to_use = ["POM", "SAG", "RPI", "BPI"]
        massey_filtered = massey_df[
            massey_df["SystemName"].isin(systems_to_use)
        ]
        massey_last = (
            massey_filtered
            .sort_values("RankingDayNum")
            .groupby(["Season", "TeamID", "SystemName"])
            .last()
            .reset_index()
        )
        # Pivot systems into columns
        massey_pivot = massey_last.pivot(
            index=["Season", "TeamID"],
            columns="SystemName",
            values="OrdinalRank"
        ).reset_index()
        massey_pivot.columns.name = None

        # Rename columns
        massey_pivot = massey_pivot.rename(columns={
            "POM": "season_POM_rank",
            "SAG": "season_SAG_rank",
            "RPI": "season_RPI_rank",
            "BPI": "season_BPI_rank",
            "Season_x": "Season"
        })
        pairs_a = set(
            zip(season_aggs["Season"], season_aggs["TeamID"]))

        pairs_b = set(
            zip(massey_pivot["Season"], massey_pivot["TeamID"]))

        common_pairs = pairs_a.intersection(pairs_b)

        season_aggs = season_aggs.merge(massey_pivot, on=["TeamID", "Season"], how="left", indicator=True)


    if elo_df is not None:
        season_aggs = season_aggs.merge(elo_df, on=["TeamID", "Season"], how="left")
    if sos_df is not None:
        season_aggs = season_aggs.merge(sos_df, on=["TeamID", "Season"], how="left")

    # Clean up
    season_aggs = season_aggs.replace([np.inf, -np.inf], np.nan)

    return season_aggs

def build_hist_seed_wp(twoteam_level, teamid_seed):
    """
    Historical win probability based on SeedDiff.
    Uses NCAA games from twoteam_level.
    """
    print(teamid_seed.columns)
    print(twoteam_level.columns)

    teamid_seed["num_seed"] = teamid_seed["Seed"].apply(seed_to_features)
    # Merge seeds into game rows
    twoteam_level = twoteam_level.merge(
        teamid_seed[["Season", "TeamID", "num_seed"]],
        on=["Season", "TeamID"],
        how="left"
    )

    seed_opp = teamid_seed.rename(
        columns={"TeamID": "OpponentID",
                 "num_seed": "num_seed_opp"}
    )

    twoteam_level = twoteam_level.merge(
        seed_opp[["Season", "OpponentID", "num_seed_opp"]],
        left_on=["Season", "OpponentID"],
        right_on=["Season", "OpponentID"],
        how="left"
    )
    print(twoteam_level.columns)

    # Compute seed difference
    twoteam_level["SeedDiff"] = (
        twoteam_level["num_seed"] -
        twoteam_level["num_seed_opp"]
    )

    # Win column already exists
    hist_seed_wp = (
        twoteam_level.groupby("SeedDiff")["Win"]
        .mean()
        .to_dict()
    )
    pd.DataFrame(
        list(hist_seed_wp.items()),
        columns=["SeedDiff", "HistSeedWP"]
    ).to_csv("seeddiff_to_HistSeedWP.csv", index=False)
    # return a dict (map) {-15.0: 0.989010989010989, -14.0: 1.0, -13.0: 0.9336734693877551 etc.


    twoteam_level["HistSeedWP"] = (
        twoteam_level["SeedDiff"]
        .map(hist_seed_wp)
    )
    twoteam_level["HistSeedWP"] = (
        twoteam_level["HistSeedWP"]
        .fillna(0.5)
    )

    return twoteam_level


MRegulargames = pd.read_csv("data/MRegularSeasonDetailedResults.csv")
MNCAAgames = pd.read_csv("data/MNCAATourneyCompactResults.csv")
teamid_seed = pd.read_csv("data/MNCAATourneySeeds.csv")
MTeams = pd.read_csv("data/MTeams.csv")
MMasseyOrdinals = pd.read_csv("data/MMasseyOrdinals.csv")

if __name__ == "__main__":
    print(MRegulargames.shape)
    #print(MMasseyOrdinals)
    #MRegularSeasonDetailedResults = pd.read_csv("data/MRegularSeasonDetailedResults.csv")
    #MRegularSeasonDetailedResults = pd.read_csv("data/MRegularSeasonDetailedResults.csv")
    #MRegularSeasonDetailedResults = pd.read_csv("data/MRegularSeasonDetailedResults.csv")
    #MRegularSeasonDetailedResults = pd.read_csv("data/MRegularSeasonDetailedResults.csv")
    #MRegularSeasonDetailedResults = pd.read_csv("data/MRegularSeasonDetailedResults.csv")


    """ ################## Double the Regular games so that we can count the
    pd[team][season][seasontype] stats with groupby
    In this way every match is listed as a win and as a loss.
    But every team can query its matches only using the first TeamID slot #################### """

    winrows, loserows = create_winrows_loserows(MRegulargames)
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



    teamid_seed["num_seed"] = teamid_seed["Seed"].apply(seed_to_features)
    twoteam_level = build_hist_seed_wp(twoteam_level, teamid_seed)
    print(twoteam_level.columns)
    team_season_stats = compute_seasonal_stats(twoteam_level, MMasseyOrdinals)

    team_season_stats.to_csv("perteam_perseason_stats.csv")
    """ ####################################################### """

    #print(team_season_stats)
