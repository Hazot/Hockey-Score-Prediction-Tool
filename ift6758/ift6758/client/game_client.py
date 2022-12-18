import pandas as pd
import numpy as np
# import src.features.tidy_dataset
# import src.data.fetch_data


class GameClient:
    def __init__(self) -> None:
        
        self.final = False
    
    # find some way to keep tract of the last event or where the last pinged stopped
    
    def feature_extract():
        # takes one game id
        
        # returns clean dataframe of the useful features
        data = fetch_game_data()
        
        pass
    
    def live_game():
        # check last eventID
        # takes the diff 
        pass


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
