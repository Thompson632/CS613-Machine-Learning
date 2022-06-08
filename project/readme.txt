CS-613 Machine Learning
Final Project - NCAAB Bracket Predictor Using Machine Learning
Kyle Coughlin, Josh Geller, Brendan Goldsmith, Robert Thompson

Versions:
- Python: 3.9.9
- pip: 22.0.4
- numpy: 1.22.3
- sportsipy: 0.6.0
- pandas: 1.4.2
- requests: 2.27.1
- beautifulsoup4: 4.10.0

Description:
- In project.py, there are two methods at the end of the file that execute each of the two main objectives we are trying to evaluate:
  - run_classifiers(file_path=file_path, game_fields=game_fields)
  - run_brackets(file_path=file_path, fields=fields, game_fields=game_fields, years=years)

NOTE: In order to build and run the brackets, you will need to have an internet connection in order to make an API call to get the specified 
years bracket data.

Each of the methods take in a file path, game fields, and years for the bracket:
- file_path: The location of the dataset
- game_fields: The chosen features we have deemed to have the most effect on the overall outcome of a college basketball game
- years: This is a list of years that we use to create and run different March Madness brackets to determine which model has the 
best accuracy

How to Run:
- From the directory where this zip file was unzipped, run the following command:
python3 project.py

Miscellaneous Notes:
- Operating System: Mac
- Architecture: 32 bit
- Virtual Python Environment: No