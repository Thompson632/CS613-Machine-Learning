from sportsipy.ncaab.teams import Team
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re


class Bracket():
    def __init__(self, year, model, fields, game_fields):
        self.year = year
        self.bracket, self.winners_bracket = self.get_bracket()
        self.teams = {}
        self.model = model
        self.correct_count = 0
        self.region_count = 0
        self.points = 0
        self.fields = fields
        self.game_fields = game_fields

    def determine_bracket_order(self, soup_data, bracket, winners_bracket):
        final_four_teams = []
        div = soup_data.findAll(id="national")
        for x in div:
            for a in x.find_all("a"):
                team = re.findall("\/cbb\/schools\/\D+\/", str(a))
                if team:
                    school = team[0]
                    school = school.replace('/cbb/schools/', '')
                    school = school.replace('/', '')
                    if school.upper() not in final_four_teams:
                        final_four_teams.append(school.upper())
        key_order = []
        for team in final_four_teams:
            for key in bracket:
                teams = bracket.get(key)
                if team in teams:
                    key_order.append(key)
        bracket = {k: bracket[k] for k in key_order}
        key_order.append("FINAL FOUR")
        winners_bracket = {k: winners_bracket[k] for k in key_order}
        return bracket, winners_bracket

    def get_winners(self, soup_data):
        winners = []
        bracket = {}
        # oregon was deemed a winner in 2021 by forfeit since vcu had covid
        # sports reference doesnt recognize them as a winner online
        append_oregon = False
        if self.year == 2021:
            append_oregon = True
        div = soup_data.find_all("div", {"class": "winner"})
        for x in div:
            for a in x.find_all("a"):
                team = re.findall("\/cbb\/schools\/\D+\/", str(a))
                if team:
                    school = team[0]
                    school = school.replace('/cbb/schools/', '')
                    school = school.replace('/', '')
                    winners.append(school.upper())
                    if school.upper() == "KANSAS" and append_oregon == True:
                        winners.append("OREGON")
                        append_oregon = False
        split = [winners[i:i + 15] for i in range(0, len(winners), 15)]
        for index, region in enumerate(split):
            if len(region) > 3:
                bracket[index+1] = region
            else:
                bracket["FINAL FOUR"] = region
        return bracket

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
        output[1] = split[0]
        output[2] = split[1]
        output[3] = split[2]
        output[4] = split[3]
        winners_bracket = self.get_winners(soup_data)
        output, winners_bracket = self.determine_bracket_order(
            soup_data, output, winners_bracket)

        return output, winners_bracket

    def run_bracket(self):
        final_four = []
        for region in self.bracket:
            self.run_region(region)
            final_four.append(self.bracket.get(region)[0])
            print("\tCorrectly Predicted Games in Region #{}: {}".format(
                region, self.region_count))
            percentage = (self.region_count/15) * 100
            print("\tPercentage of Correctly Predicted Games in Region #{}: {}".format(
                region, percentage))
            self.region_count = 0
            print("")

        self.bracket.clear()
        self.bracket["FINAL FOUR"] = final_four
        self.run_region("FINAL FOUR", 160)
        print("\tCorrectly Predicted Games in FINAL FOUR: {}".format(
            self.region_count))
        percentage = (self.region_count/3) * 100
        print("\tPercentage of Correctly Predicted Games in FINAL FOUR: {}".format(
            percentage))
        print("")

        print("\t{} Champion: {}".format(
            self.year, self.bracket["FINAL FOUR"][0]))
        print("\tCorrectly Predicted Games: {}".format(self.correct_count))
        percentage = (self.correct_count/63) * 100
        print("\tPercentage of Correctly Predicted Games: {}".format(percentage))
        print("\tTotal Points: {}".format(self.points))

    def run_region(self, region, starting_points=10):
        winners = []
        region_size = len(self.bracket.get(region))
        winners_bracket = self.winners_bracket.get(region)
        print(region)
        while region_size != 1:
            for index, team in enumerate(self.bracket.get(region)):
                if index % 2 == 0:
                    actual_winner = winners_bracket.pop(0)
                    home = team
                    away = self.bracket.get(region)[index+1]
                    game = self.create_game(
                        away, home, self.fields, self.game_fields[:-1])
                    pred = self.model.predict(game)[0]
                    if pred == 1:  # home win
                        winners.append(home)
                        if home == actual_winner:
                            self.correct_count += 1
                            self.region_count += 1
                            self.points += starting_points
                    else:  # away win
                        winners.append(away)
                        if away == actual_winner:
                            self.correct_count += 1
                            self.region_count += 1
                            self.points += starting_points
                    print("\tH: {} vs A: {}, home win = {}, actual winner = {}".format(
                        home, away, pred, actual_winner))
            region_size = len(winners)
            self.bracket[region] = winners
            winners = []
            starting_points = starting_points * 2
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