# import pandas as pd
# import numpy as np

# from ift6758.data import pd_make_df

# class GameClient:
#     def __init__(self) -> None:
#         self.gameId = 0
#         self.last_eventIdx = None
#         self.game_ended = None
        
        
#     def process_query(self, gameId, return_raw=False, model_name="xgboost-base-all-features"):
#         """
#         Produces a dataframe with the features required by the XGBoost model.
#         Returns None if the game has already been fully processed, or if the updated
#             records do not contain data (e.g. breaks, period changes, etc.)
#         Returns a dataframe if valid records are found.

#         Model names:
#         "xgboost-best-select-features"
#         "xgboost-best-all-features"
#         "xgboost-base-all-features"
#         """
#         full_model = True
#         if model_name == "xgboost-best-select-features":
#             full_model = False
        
#         game_year = int(str(gameId)[:4])
#         use_cache = True if game_year < 2022 else False
#         # game has already been fully processed
#         if self.gameId == gameId and self.game_ended: return None 
        
#         # load game
#         ##
#         df = pd_make_df.create_dataframe_from_game(gameId, use_cache=use_cache)
        
#         # if same game, slice. last_eventIdx was set in the last call
#         if self.gameId == gameId:
#             df = df[df["eventIdx"] > self.last_eventIdx] 
            
#         self.game_ended = "GAME_END" in df["eventType"].values
#         self.last_eventIdx = df.iloc[-1].eventIdx
        
#         filtered = pd_make_df.full(df).reset_index(drop=True)
        
#         returned_df = pd_make_df.aug3(filtered, full_model)
# #         returned_df["team"] = filtered["teamTriCode"]
#         returned_df.insert(  # same as the line above but moving team to coln #1
#             loc=0, column="team",
#             value=filtered["teamTriCode"]
#         )
        
#         if len(returned_df) == 0: return None
#         else: return df if return_raw else returned_df

# if __name__ == "__main__":
    
#     # STREAMLIT APPLICATION
#     game = GameClient()
#     game.live_game()
    
#     from ift6758.client.serving_client import ServingClient
#     serv = ServingClient()
    
#     currentGameID = 2021020329 # fetched from streamlit user input
#     if game.gameID != currentGameID:
#         game = GameClient(currentGameID)
    
#     X = game.feature_extract()
    
#     output = serv.predict(X)
    
#     serv.features

import pandas as pd
import numpy as np

from ift6758.data import pd_make_df

class GameClient:
    def __init__(self) -> None:
        self.gameId = 0
        self.last_eventIdx = None
        self.game_ended = None
        self.model_name = None
        
        
    def process_query(self, gameId, return_raw=False, model_name="xgboost-base-all-features"):
        """
        Produces a dataframe with the features required by the XGBoost model.
        Returns None if the game has already been fully processed, or if the updated
            records do not contain data (e.g. breaks, period changes, etc.)
        Returns a dataframe if valid records are found.

        Model names:
        "xgboost-best-select-features"
        "xgboost-best-all-features"
        "xgboost-base-all-features"
        """
        full_model = True
        if model_name == "xgboost-best-select-features":
            full_model = False
        
        game_year = int(str(gameId)[:4])
        use_cache = True if game_year < 2022 else False
        if self.model_name != model_name: self.gameId = 0
        self.model_name = model_name
        # game has already been fully processed
        if self.gameId == gameId and self.game_ended: return None 
        
        # load game
        df = pd_make_df.create_dataframe_from_game(gameId, use_cache=use_cache)
        
        # if same game, slice. last_eventIdx was set in the last call
        if self.gameId == gameId:
            df = df[df["eventIdx"] > self.last_eventIdx] 
            
        self.game_ended = "GAME_END" in df["eventType"].values
        self.last_eventIdx = df.iloc[-1].eventIdx
        
        filtered = pd_make_df.full(df).reset_index(drop=True)
        
        returned_df = pd_make_df.aug3(filtered, full_model)
#         returned_df["team"] = filtered["teamTriCode"]
        returned_df.insert(  # same as the line above but moving team to coln #1
            loc=0, column="team",
            value=filtered["teamTriCode"]
        )
        self.gameId = gameId
        if len(returned_df) == 0: return None
        else: return df if return_raw else returned_df

if __name__ == "__main__":
    
    # STREAMLIT APPLICATION
    game = GameClient()
    game.live_game()
    
    from ift6758.client.serving_client import ServingClient
    serv = ServingClient()
    
    currentGameID = 2021020329 # fetched from streamlit user input
    if game.gameID != currentGameID:
        game = GameClient(currentGameID)
    
    X = game.feature_extract()
    
    output = serv.predict(X)
    
    serv.features
