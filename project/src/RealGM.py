import requests
import pandas as pd
from bs4 import BeautifulSoup
import math_util
from datetime import date
import re
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def get_stats(url,team):
    r=requests.get(url)
    soup=BeautifulSoup(r.text,'html.parser')
    table=soup.select_one(".tablesaw ")
    df=pd.read_html(str(table))[0]
    return df.loc[df['Team'] == team]

def get_todays_games():
    r=requests.get("https://basketball.realgm.com/ncaa/")
    soup=BeautifulSoup(r.text,'html.parser')
    table = soup.find('table', attrs={'class':'basketball force-table'})
    table_body = table.find('tbody')
    data = []
    rows = table_body.find_all('tr')
    today = date.today()
    day = str(today.strftime('%b')+". "+str(today.day))
    for row in rows:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        game = [ele for ele in cols if ele] # Get rid of empty values
        if day in game[0]:
            if " (L)" not in game[1] and " (W)" not in game[1]:
                away = re.sub("\#\d{1,}\s", '', game[1])
                home = re.sub("\#\d{1,}\s", '', game[2] )
                data.append((away,home)) 
    return data

def get_team(team):
    base_averages = get_stats("https://basketball.realgm.com/ncaa/team-stats/2023/Averages/Team_Totals",team)
    averages = base_averages[["Team","3P%","FT%"]]
    advanced = get_stats("https://basketball.realgm.com/ncaa/team-stats/2023/Advanced_Stats/Team_Totals/",team)[["Team","ORtg","eFG%","TRB%","TOV%","TS%"]]
    rates = base_averages[["FTA","FGA","3PA"]]
    free_throw_rate = rates["FTA"] / rates["FGA"]
    three_point_rate = rates["3PA"] / rates["FGA"]
    averages.loc[:,'free_throw_attempt_rate'] = free_throw_rate
    averages.loc[:,'three_point_attempt_rate'] = three_point_rate

    merged = pd.merge(averages,advanced,how="outer",on="Team")
    merged.rename(columns={'3P%': 'three_point_field_goal_percentage', 'FT%': 'free_throw_percentage', 
                           'ORtg': 'offensive_rating', 'eFG%': 'effective_field_goal_percentage',
                           'TRB%': 'total_rebound_percentage', 'TOV%': 'turnover_percentage',
                           'TS%': 'true_shooting_percentage'}, inplace=True)
    merged = merged.iloc[:,[6,3,2,5,4,1,7,9,8]]
    return merged

def get_game(home_df,away_df,game_fields,means,stds):
    #need dummy column to join on, will remove later on in method
    away_df['joincol'] = 1
    home_df['joincol'] = 1
    game = pd.DataFrame(columns=game_fields)
    game = pd.merge(away_df, home_df, how='outer', on='joincol', suffixes=('_away', '_home'))
    game = game.drop(columns=['joincol']).dropna(axis='columns')
    game = game.to_numpy()
    return math_util.z_score_data(game, means, stds)
