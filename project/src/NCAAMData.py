from sportsreference.ncaab.boxscore import Boxscores
from sportsreference.ncaab.schedule import Schedule
from datetime import datetime
from sportsreference.ncaab.teams import Team
from sportsreference.ncaab.teams import Teams
import pandas as pd
from datetime import timedelta
import requests
class NCAAMData:
    def __init__(self,year):
        self.date = datetime.today()
        self.year = year
        self.teams = Teams(year)
        print(self.teams)
    
    def get_todays_games(self):
        ranked= {}
        unranked = {}
        dict = {}
        num = 1
        print("Getting games for: "+str(datetime.today()- timedelta(days = 1)))
        games_today = Boxscores(datetime.today()- timedelta(days = 1))
        for today in games_today.games.values():
            for game in today:
                away_abbr = game["away_abbr"].upper()
                home_abbr = game["home_abbr"].upper()
                away_rank = game["away_rank"]
                home_rank = game["home_rank"]
                #only adding games with a ranked team
                if  home_rank != None or away_rank != None:
                    ranked[num] = self.create_game(away_abbr,home_abbr)
                    num += 1
        dict["ranked"] = ranked
        print("Number of games: "+str(num))
        return dict
    
    def create_game(self,away_abbr,home_abbr):
        try:
            game = []
            game.append(away_abbr)
            game.append(home_abbr)
        except:
            print("Error with " + away_abbr + " vs " + home_abbr + " game thrown out")
        return game
    
    def create_game_dataframe(self,away_abbr,home_abbr, team_fields, game_fields):
        away = Team(team_name=away_abbr,year=self.year)
        away_df = away.dataframe[team_fields].dropna(axis='columns')
        home = Team(team_name=home_abbr,year=self.year)
        home_df = home.dataframe[team_fields].dropna(axis='columns')

        #need dummy column to join on, will remove later on in method
        away_df['joincol'] = 1
        home_df['joincol'] = 1
        game = pd.DataFrame(columns=game_fields)
        game = pd.merge(away_df, home_df, how='outer', on='joincol', suffixes=('_away', '_home'))
        game = game.drop(columns=['joincol']).dropna(axis='columns')
        return game