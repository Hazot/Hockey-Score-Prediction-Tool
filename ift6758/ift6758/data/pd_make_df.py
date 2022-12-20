import pandas as pd
import numpy as np
import json
import os

from sklearn.preprocessing import OneHotEncoder

from ift6758.data.fetch_data import FetchData
pd.options.mode.chained_assignment = None

class RinkMapping:
    def __init__(self, teamType):
        """
        give teamType that matches period 1 and is on the right.
        """
        if teamType not in ("home", "away"): 
            raise Exception("Not an allowed input")
        self.ref = teamType
        self.val_map = {1: "right", -1: "left"}
        
    def pred(self, period, teamType):
        period = int(period)
        if period % 2 == 1:
            res = 1 if teamType == self.ref else -1
        else:
            res = -1 if teamType == self.ref else 1

        return self.val_map[res]
def add_rinkSide(df, data):
    # Rink Side
    home_away = {data["gameData"]["teams"][teamType]["triCode"]: teamType for teamType in ("away", "home")}
    df["teamType"] = df["teamTriCode"].map(home_away)
    infer_rinkSide = False
    try:
        rinkside_df = pd.json_normalize(data["liveData"]["linescore"]["periods"])[["num", "home.rinkSide", "away.rinkSide"]]
        rinkside_df.columns = ["period", "home", "away"]
        rinkside_df = rinkside_df.melt("period", value_name="rinkSide", var_name="teamType")

        df = pd.concat([df.set_index(["period", "teamType"]), rinkside_df.set_index(["period", "teamType"])], axis=1).reset_index()
    except:
        infer_rinkSide = True
        df["rinkSide"] = None

    # Infer rinkSide if none was found
    if infer_rinkSide:
        infer = df[df.period < 4].copy()

        x_mean = (infer[infer.rinkSide.isnull()]
                  .groupby(["period", "teamType"])["coordinate_x"]
                  .mean()
        )
        mins = (
            x_mean
            .reset_index()
            .groupby(["period"])["coordinate_x"]
            .idxmin()
            .values
        )

        
        merge_rightRink = (
            x_mean
            .reset_index()
            .loc[mins]
            .set_index(["period", "teamType"])
        )
        merge_rightRink["rinkSide_app"] = "right"
        merge_rightRink = merge_rightRink[["rinkSide_app"]]
        merge_rightRink = merge_rightRink.reset_index()

        rinkmap = RinkMapping(merge_rightRink.query("period == 1")["teamType"].values)

        df["rinkSide"] = df[["period", "teamType"]].apply(lambda x: rinkmap.pred(*x), axis=1)

    return df


def add_features_set1(df):
    df[["adjusted_x", "adjusted_y"]] = df[["coordinate_x", "coordinate_y"]].copy()
    df.loc[df.rinkSide == "right", ["adjusted_x", "adjusted_y"]] *= -1
    
    df["distanceFromGoal"] = np.sqrt(
        (df["adjusted_x"] - 89)**2 + (df["adjusted_y"])**2
    )
    df["shotAngle"] = np.arcsin(df["adjusted_y"]/df["distanceFromGoal"])
    
    # Time convert
    df["periodTimeSec"] = (pd.to_datetime(df["periodTimeSec"], format="%M:%S") - pd.to_datetime("0:0", format="%M:%S")).dt.total_seconds()
    
    #shifted df for differencing
    shifted_df = df.shift()
    
    # Feature Eng 2
    df["rebound"] = shifted_df["eventType"] == "SHOT"
    df["lastEventType"] = shifted_df["eventType"]
    df[['lastEventCoord_x', 'lastEventCoord_y']] = shifted_df[["coordinate_x", "coordinate_y"]]

    df["timeDifference"] = df["periodTimeSec"] - shifted_df["periodTimeSec"]

    df["distanceDifference"] = np.sqrt(
        (df["coordinate_x"] - shifted_df["coordinate_x"])**2 + 
        (df["coordinate_y"] - shifted_df["coordinate_y"])**2
    )

    df["shotAngleDifference"] = np.where(
        df["lastEventType"] == "SHOT", 
        df["shotAngle"] - shifted_df["shotAngle"], 
        0
    )

    df["speed"] = np.abs(df["distanceDifference"]/df["timeDifference"])
    df["speed"] = df['speed'].replace([np.inf, -np.inf], -1) # in case of division by 0
    
    df = df.drop(["adjusted_x", "adjusted_y"], axis="columns")
    
    return df


def parse_players(players_list):
    shooter = None
    goalie  = None
    
    
    for player in players_list:
        if player["playerType"] in ("Shooter", "Scorer"):
            shooter = player["player"]["fullName"]
        if player["playerType"] == "Goalie":
            goalie = player["player"]["fullName"]
    if shooter is None and goalie is None: return None
    return f"{shooter} | {goalie}"

def add_players(df):
    df = df.dropna(subset="players")
    df["parsed_players"] = df["players"].map(parse_players)
    df[["shooter", "goalie"]] = df["parsed_players"].str.split("|", expand=True)
    df = df.drop(["parsed_players", "players"], axis=1)
    return df


def create_dataframe_from_game(game_id:str, use_cache=True):
    helper = FetchData()
    data = helper.get_play_by_play(game_id, use_cache=use_cache)
        
    allPlays_columns = [
    #     'result.event', 'result.eventCode', 
        'result.eventTypeId', # 'result.description', 
        'about.eventIdx', # 'about.eventId', 
        'about.period', 'about.periodType', # 'about.ordinalNum', 
        'about.periodTime', 'about.periodTimeRemaining',
        #'about.dateTime', 'about.goals.away', 'about.goals.home', 
        'players', 'coordinates.x', 'coordinates.y',
        # 'team.id', 'team.name', 'team.link', 
        'team.triCode',
        'result.secondaryType', # 'result.penaltySeverity', 'result.penaltyMinutes', 
        # 'result.strength.code', # 'result.strength.name','result.gameWinningGoal', 
        'result.emptyNet'
    ]
    allPlays_columns_map = {
        'result.eventTypeId'       : "eventType", 
        'about.eventIdx'           : "eventIdx", 
        'about.period'             : "period", 
        'about.periodType'         : "periodType", 
        'about.periodTime'         : "periodTimeSec", 
        'about.periodTimeRemaining': "periodTimeRem",
        'players'                  : "players", 
        'coordinates.x'            : "coordinate_x", 
        'coordinates.y'            : "coordinate_y",
        'result.secondaryType'     : "shotType",
        'team.triCode'             : "teamTriCode",
        # 'result.strength.code'     : "shotStrength", 
        'result.emptyNet'          : "emptyNet"
    }
    
    #load exceptions
    try:
        df = pd.json_normalize(data["liveData"]["plays"]["allPlays"])
    except: return None
    if df.empty: return None 
    
    if "result.emptyNet" not in df.columns:
        df["result.emptyNet"] = None
    
    
    df = df[allPlays_columns]
    df = df.rename(allPlays_columns_map, axis=1)
    df["gameId"] = game_id
    
    #rinkSide
    df = add_rinkSide(df, data)
    df = add_features_set1(df)
#     df = add_players(df)
    df["emptyNet"] = df.emptyNet.fillna(False)
    df["emptyNet"] = df.emptyNet.astype(int)

    # Add Is goal column to dataframe efficiently
    df["isGoal"] = df["eventType"] == "GOAL"
    df["isGoal"] = df.isGoal.astype(int)
    
    return df[[
        "gameId",
        "eventType", 
        "eventIdx", 
        "period", 
        "periodType", "periodTimeSec", "periodTimeRem",
#         "shooter", "goalie",
#         "players", 
        "coordinate_x", "coordinate_y",
        "shotType",
        "teamTriCode",
        # "shotStrength", 
        "emptyNet",
        "isGoal",
        "rinkSide",
        "distanceFromGoal",
        "shotAngle",
        "rebound",
        "lastEventType", "lastEventCoord_x", "lastEventCoord_y",
        "timeDifference", "distanceDifference", "shotAngleDifference",
        "speed",
    ]]

def full(df):
    return df[
        (df.eventType.isin(["SHOT", "GOAL"]))
    ]

def aug1(df):
    return full(df)[['emptyNet', 'distanceFromGoal', 'shotAngle', 'isGoal']]

def aug2(df, call_full=True):
    """
    added call_full to avoid breaking things
    """
    colns = [
        'periodTimeSec', 'period', 'coordinate_x', 'coordinate_y', 
        'distanceFromGoal', 'shotAngle', 'shotType', 'lastEventType',
        'lastEventCoord_x', 'lastEventCoord_y', 'timeDifference',
        'distanceDifference', 'rebound', 'shotAngleDifference', 'speed',
        'isGoal'
    ]
    
    if call_full:
        return full(df)[colns]
    else:
        return df[colns]

def aug3(df, full_model):
    """
    Added to directly preprocess into XGBoost compatible data.
    Note: individual games don't map out the support of 'shotType' and 'lastEventType'
    which means that certain values will not be orthogonalized, and therefore needs to be added 
    manually to cover for the one hot encoder.
    """
    df = aug2(df, call_full=False)
    df = df.dropna().reset_index(drop=True)

    enc_style = OneHotEncoder()
    
    #shotType
    enc_results = enc_style.fit_transform(df[["shotType"]])
    enc_df = pd.DataFrame(enc_results.toarray(), columns=enc_style.categories_[0])
    df = df.join(enc_df).drop(columns="shotType", errors="ignore")
    
    # lastEventType
    enc_results = enc_style.fit_transform(df[["lastEventType"]])
    enc_df = pd.DataFrame(enc_results.toarray(), columns=enc_style.categories_[0])
    df = df.join(enc_df).drop(columns='lastEventType', errors='ignore')

    df["rebound"] = df["rebound"].astype(int)
    
    shotType_support = [
        'Backhand', 'Deflected', 'Slap Shot', 
        'Snap Shot', 'Tip-In', 'Wrap-around', 
        'Wrist Shot']
    
    lastEventType_support = [
        'BLOCKED_SHOT', 'FACEOFF', 'GIVEAWAY',
       'GOAL', 'HIT', 'PENALTY', 'MISSED_SHOT', 'SHOT', 
        'TAKEAWAY']
    
    for col in shotType_support:
        if col not in df.columns:
            df[col] = 0
    
    for col in lastEventType_support:
        if col not in df.columns:
            df[col] = 0
    
    df = df[['isGoal', 'periodTimeSec', 'period', 'coordinate_x', 'coordinate_y',
       'distanceFromGoal', 'shotAngle', 'lastEventCoord_x', 'lastEventCoord_y',
       'timeDifference', 'distanceDifference', 'rebound',
       'shotAngleDifference', 'speed', 'Backhand', 'Deflected', 'Slap Shot',
       'Snap Shot', 'Tip-In', 'Wrap-around', 'Wrist Shot', 'BLOCKED_SHOT',
       'FACEOFF', 'GIVEAWAY', 'GOAL', 'HIT', 'MISSED_SHOT', 'PENALTY', 'SHOT',
       'TAKEAWAY']]
    
    if full_model:
        return df
    
    # isGoal should not have been there at any point
    X = df.drop(columns=['PENALTY', 'lastEventCoord_x'], errors='ignore')
    
    return X