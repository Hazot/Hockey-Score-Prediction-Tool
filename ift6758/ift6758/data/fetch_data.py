import pandas as pd

import requests
import json
import os
import re


class FetchData:
    def __init__(self):
        pass

    def format_game_id(self, year: int, game_type: str, game_number=None, round_number=None, matchup=None, game=None):
        """
        This method is more general and returns the gameID format for any specified game type.
        Can to generate gameID for one specific game if given game_number for pre/regular season, or round/matchup/game for playoffs
        Otherwise generates and returns a list of all the Game IDs for play-by-play data from NHL's API,
        for the <year/year+1> <game_type>, according to the format documented
        in https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md#game-ids
        i.e. YYYY02####, where YYYY = year, 0i = {1,2,3,4} for game_type, #### = game number
        Args:
            year (int): The first year of the season to retrieve, i.e. for the 2016-17
            season you'd put in 2016
            game_type (str): The game type of the game(s) ID(s) to format in [preseason, regular season, playoffs, all-star]
            game_number: for preseason and regular season this is the specific game number
            round: for playoffs only, this is the round of the match
            matchup: for playoffs only, this is the matchup from the round
            game: for playoffs only, this the game in the round and matchup given
        Returns:
            IDs: a list of all the game IDs for this specific year's game type or a specific game if keyword arguments are specified
        """
        # if game_type is pre-season
        if bool(re.search(r"pre\s?-?season", game_type.lower())):
            if game_number is not None:
                return str(year) + "01" '{:04d}'.format(game_number)
            else:
                return self.game_ids_pre(year)

        # if game_type is regular season
        if bool(re.search(r"regular\s?-?season", game_type.lower())):
            if game_number is not None:
                return str(year) + "02" '{:04d}'.format(game_number)
            else:
                return self.game_ids_regular(year)

        # if game_type is playoffs
        elif bool(re.search(r"play-?offs?", game_type.lower())):
            if round_number is not None:
                try:
                    return str(year) + "03" + "0" + str(round_number)+ str(matchup) + str(game)
                except ValueError:
                    print("Need to specify: round, matchup and game")
            else:
                return self.game_ids_playoff(year)

        # if game_type is all-star (this is one game only)
        elif bool(re.search(r"all-?\s?stars?", game_type.lower())):
            return str(year) + "04" + "0000"

        # if game_type is none of the above, raise exception
        else:
            raise ValueError('Valid game types are: regular season, preseason, playoffs, all-star')

    def game_ids_pre(self, year: int):
        """
        Generate and returns a list of all the Game IDs for play-by-play data from NHL's API,
        for the <year/year+1> PRESEASON, according to the format documented
        in https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md#game-ids
        i.e. YYYY02####, where YYYY = year, 01 = regular season, #### = specific game number
        note: #### in (0001, 1271) from 2017 and onwards, and (0001, 1230) otherwise
        Args:
            year (int): The first year of the season to retrieve, i.e. for the 2016-17
            season you'd put in 2016
        Returns:
            IDs: a list of all the game IDs for this specific year's regular season
        """
        if year < 2017:
            n = 1230
        else:
            n = 1271
        IDs = [str(year) + "01" + '{:04d}'.format(i) for i in range(1, n+1)]
        return IDs

    def game_ids_regular(self, year: int):
        """
        Generate and returns a list of all the Game IDs for play-by-play data from NHL's API,
        for the <year/year+1> REGULAR SEASON, according to the format documented
        in https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md#game-ids
        i.e. YYYY02####, where YYYY = year, 02 = regular season, #### = specific game number
        note: #### in (0001, 1271) from 2017 and onwards, and (0001, 1230) otherwise
        Args:
            year (int): The first year of the season to retrieve, i.e. for the 2016-17
            season you'd put in 2016
        Returns:
            IDs: a list of all the game IDs for this specific year's regular season
        """
        if year < 2017:
            n = 1230
        elif year == 2020:
            n = 868
        else:
            n = 1271
        IDs = [str(year) + "02" + '{:04d}'.format(i) for i in range(1, n+1)]
        return IDs


    def game_ids_playoff(self, year: int):
        """
        Generate and returns a list of all the Game IDs for play-by-play data from NHL's API,
        for the <year/year+1> PLAYOFFS, according to the format documented
        in https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md#game-ids
        i.e. YYYY020RMG, where YYYY = year, 03 = playoffs, 0RMG = specific game number
        note: R = round, M = Matchup, G = Game (out of 7)
        Args:
            yea (int): The first year of the season to retrieve, i.e. for the 2016-17
            season you'd put in 2016
        Returns:
            IDs: a list of all the game IDs for this specific year's playoff
        """
        matchups = [8, 4, 2, 1]
        IDs = [(str(year) + "030" + str(R) + str(M) + str(G))
               for R in range(1, 5) for M in range(1, matchups[R-1]+1) for G in range(1, 8)]
        if year == 2017 or year == 2020:
            return IDs[:-2]
        elif year == 2016 or year == 2019 or year == 2021:
            return IDs[:-1]
        return IDs

    def get_play_by_play(self, 
                         game_id: str, 
                         path=os.path.abspath(os.path.join(os.path.dirname(__file__),"../../../data/raw/")), 
                         return_data=True, 
                         use_cache=True):

        """
        If file doesn't exist: Downloads play-by-play data for the give game ID and returns the raw json content
        If file already exists: Skips download and returns the raw json content
        The file is saved in json format at path/<game_id>
        Args:
            game_id (str): Game ID for play-by-play data according to the format documented
                in https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md#game-ids
            path: Path to which the file is saved
            use_cache: whether we check for existence of file.
        Returns: raw data for the specific game in the form of a json object
        """
        
        # Check if data has already been fetched in
        if not os.path.exists("{}/{}.json".format(path, game_id)):
            # If file doesn't exist, download data using requests package
            # Store at path/<game_id> in json format
            response = requests.get("https://statsapi.web.nhl.com/api/v1/game/{}/feed/live/".format(game_id))
            json_response = response.json()

            with open('{}/{}.json'.format(path, game_id), "w") as f:
                json_str = json.dumps(json_response)
                f.write(json_str)
        else:
            # file exists. but we don't want to use cached.
            if not use_cache:
                response = requests.get("https://statsapi.web.nhl.com/api/v1/game/{}/feed/live/".format(game_id))
                json_response = response.json()

                with open('{}/{}.json'.format(path, game_id), "w") as f:
                    json_str = json.dumps(json_response)
                    f.write(json_str)
                    
        if return_data:
            with open("{}/{}.json".format(path, game_id), 'r') as f:
                data = json.load(f)
            return data
    
    def download_everything(self, years: list, path=os.path.abspath(os.path.join(os.path.dirname(__file__),"../../data/raw/")), return_data=True):
        """_summary_

        Args:
            years (list): _description_
        """
        for year in years:
        
            regular_season_games = self.game_ids_regular(year)
            playoffs_games = self.game_ids_playoff(year)

            for game_id in regular_season_games:
                self.get_play_by_play(game_id, path, return_data)
            for game_id in playoffs_games:
                self.get_play_by_play(game_id, path, return_data)
    
        print('Download finished!')
    

    def get_scores(self, game_id: str, use_cache=True):
        """
        Get number of real goals from live feed dict (raw data)
        """
        data = self.get_play_by_play(game_id, use_cache=use_cache)
        goal_a = None # away team
        goal_h = None # home team
        
        if data["liveData"].get("linescore") is not None:
            goal_a = data["liveData"]["linescore"]["teams"]["away"]["goals"]
            goal_h = data["liveData"]["linescore"]["teams"]["home"]["goals"]
        
        # Bug while trying to get number of goals from linescore
        if goal_a is None and goal_h is None: 
            goal_a = data["liveData"]["plays"]["currentPlay"]["about"]["goals"]["away"]
            goal_h = data["liveData"]["plays"]["currentPlay"]["about"]["goals"]["home"]

        # Get away and home teams tricode
        away = data["gameData"]["teams"]["away"]["triCode"]
        home = data["gameData"]["teams"]["home"]["triCode"]

        real_goals = [goal_a, goal_h]
        teams = [away, home]    

        return real_goals, teams



if __name__ == "__main__":
    ### To test code ###
    fetch = FetchData()
    regular_season_games = fetch.game_ids_regular(2017)
    playoffs_games = fetch.game_ids_playoff(2017)
    data = fetch.get_play_by_play(regular_season_games[0])
    print(type(data))
    # print(data)
    for game_id in regular_season_games:
        data = fetch.get_play_by_play(game_id)
    for game_id in playoffs_games:
        data = fetch.get_play_by_play(game_id)
    
    # Test to download everything
    # years = [2015, 2016, 2017, 2018, 2019] # from 2015-2016 to 2019-2020 seasons
    # fetch.download_everything(years)
