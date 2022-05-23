from sportsipy.ncaab.teams import Team
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re


class Bracket():
    def __init__(self, year, model, fields, game_fields):
        self.year = year
        self.bracket = self.get_bracket()
        self.teams = {}
        self.model = model
        self.fields = fields
        self.game_fields = game_fields

    def get_bracket(self):
        url = 'https://www.sports-reference.com/cbb/postseason/' + \
            str(self.year)+'-ncaa.html'
        res = requests.get(url)
        soup_data = BeautifulSoup(res.text, 'html.parser')
        div = soup_data.findAll(id="bracket")
        teams = []
        for x in div:
            for a in x.find_all("a"):
                team = re.findall("\/cbb\/schools\/\D+\/", str(a))
                if team:
                    school = team[0]
                    school = school.replace('/cbb/schools/', '')
                    school = school.replace('/', '')
                    if school.upper() not in teams:
                        teams.append(school.upper())

        # breaking into regions
        split = [teams[i:i + 16] for i in range(0, len(teams), 16)]
        output = {}
        output["EAST"] = split[0]
        output["MIDWEST"] = split[1]
        output["SOUTH"] = split[2]
        output["WEST"] = split[3]

        return output

    def run_bracket(self):
        final_four = []
        
        for region in self.bracket:
            self.run_region(region)
            
        final_four.append(self.bracket["EAST"][0])
        final_four.append(self.bracket["MIDWEST"][0])
        final_four.append(self.bracket["SOUTH"][0])
        final_four.append(self.bracket["WEST"][0])
        self.bracket.clear()
        self.bracket["FINAL FOUR"] = final_four
        self.run_region("FINAL FOUR")
        
        print("{} Champion: {}".format(
            self.year, self.bracket["FINAL FOUR"][0]))

    def run_region(self, region):
        winners = []
        region_size = len(self.bracket.get(region))
        print(region)
        
        while region_size != 1:
            for index, team in enumerate(self.bracket.get(region)):
                if index % 2 == 0:
                    home = team
                    away = self.bracket.get(region)[index+1]
                    game = self.create_game(
                        away, home, self.fields, self.game_fields[:-1])
                    pred = self.model.evaluate_model(game)[0]
                    if pred == 1:  # home win
                        winners.append(home)
                    else:  # away win
                        winners.append(away)
                    print("\tH: {} vs A: {}, home win = {}".format(
                        home, away, pred))

            region_size = len(winners)
            self.bracket[region] = winners
            winners = []
            print("")

    def get_team_by_name(self, team_name, year, team_fields):
        # return a team entry from sportsipy
        team = Team(team_name=team_name, year=year)
        df = team.dataframe[team_fields].dropna(axis='columns')
        return df

    def create_game(self, away_abbr, home_abbr, team_fields, game_fields):
        # combine two team entries into one dataframe to be passed to a model
        # only call api for a team once
        if away_abbr not in self.teams:
            away_df = self.get_team_by_name(away_abbr, self.year, team_fields)
            self.teams[away_abbr] = away_df
        else:
            away_df = self.teams[away_abbr]

        if home_abbr not in self.teams:
            home_df = self.get_team_by_name(home_abbr, self.year, team_fields)
            self.teams[home_abbr] = home_df
        else:
            home_df = self.teams[home_abbr]

        # need dummy column to join on, will remove later on in method
        away_df['joincol'] = 1
        home_df['joincol'] = 1

        game = pd.merge(away_df, home_df, how='outer',
                        on='joincol', suffixes=('_away', '_home'))
        game = game.drop(columns=['joincol']).dropna(axis='columns')
        game.columns = game_fields
        return game.to_numpy()