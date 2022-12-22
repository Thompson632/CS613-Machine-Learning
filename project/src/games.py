import RealGM
import warnings
from sklearn.model_selection import train_test_split
from sklearn import metrics 
import math_util
import data_util
from random_forest import RandomForest 


warnings.filterwarnings("ignore")

class Games:
    def __init__(self):
        self.start()

    def start(self,file_path="games.csv"):
        # File Input Path
        self.file_path = file_path

        # Define our most informative features based on our knowledge
        self.fields = ['offensive_rating', 'effective_field_goal_percentage', 'total_rebound_percentage', 'free_throw_attempt_rate',
                'free_throw_percentage', 'three_point_attempt_rate', 'three_point_field_goal_percentage', 'turnover_percentage', 'true_shooting_percentage']
        # Prepend away and home to each field
        self.game_fields = self.generate_game_fields(self.fields, "home_win")
        self.random_forest,means,stds = self.random_forest(filename=file_path, forest_size=100,
                                               num_observations_per_tree=0.25,
                                               min_observation_split=2,
                                               min_information_gain=0,
                                               game_fields=self.game_fields)
        
        self.models = []
        self.models.append(self.random_forest)
        teams = []#RealGM.get_todays_games()
        teams.append(("Virginia","Miami (FL)"))
        teams.append(("Duke","Wake Forest"))
        teams.append(("Marquette","Providence"))
        
        for game in teams:
            self.predict(game,means,stds)
            
    def predict(self,teams,means,stds):
        away = RealGM.get_team(teams[0])
        home = RealGM.get_team(teams[1])
        game =  RealGM.get_game(home,away,self.game_fields,means,stds)
        if game is not None:
            for model in self.models:
                score = model.predict(game)
                print("{} vs {}, home win = {}".format(teams[0],teams[1],score[0]))
        print("")
        
    def load_data(self,filename, columns):
        data = data_util.load_data(filename, columns=columns)
        data = data_util.shuffle_data(data, 0)

        X, y = data_util.get_features_actuals(data)

        means, stds = math_util.calculate_feature_mean_std(X)
        X_zscored = math_util.z_score_data(X, means, stds)
        return X_zscored, y, means, stds
    
    def random_forest(self,filename, forest_size, num_observations_per_tree,
                            min_observation_split, min_information_gain,
                            game_fields):
        print("\n======================================================")
        print("TRAINING RANDOM FOREST MODEL FOR PREDICTION:")

        X, y, means,stds = self.load_data(filename, game_fields)

        model = RandomForest(forest_size=forest_size, num_observations_per_tree=num_observations_per_tree,
                            min_observation_split=min_observation_split,
                            min_information_gain=min_information_gain)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))        
        return model, means, stds

    def generate_game_fields(self,fields, target):
        away_columns = []
        home_columns = []

        for field in fields:
            away_columns.append("away_"+field)
            home_columns.append("home_" + field)
        result = away_columns + home_columns
        result.append(target)
        return result

games = Games()
