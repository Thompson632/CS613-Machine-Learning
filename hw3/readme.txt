CS-613 Machine Learning
Assignment 3 - Classification
Robert Thompson

Versions:
- Python: 3.9.9
- pip: 22.0.4
- numpy: 1.22.3

Description:
- In test_classification.py, there is three methods at the end of the file that execute for each of the different parts of the assignment:
  - naive_bayes
  - decision_trees
  - multi_class_classification
  
Each of the methods take the following parameter (learning rate, epochs, and stability) that can be adjusted for more accuracy. I left the the parameters
as follows because as I increased the epochs, my CPU increased exponentially but I did let it run and did notice increased accuracy for both models:
  - learning rate = 0.1
  - epochs = 10000
  - stability = 10e-7

How to Run:
- From the directory where this zip file was unzipped, run the following command:
python3 test_classification.py

Miscellaneous Notes:
- Operating System: Mac
- Architecture: 32 bit
- Virtual Python Environment: No