CS-613 Machine Learning
Assignment 3 - Classification
Robert Thompson

Versions:
- Python: 3.9.9
- pip: 22.0.4
- numpy: 1.22.3

Description:
- In test_classification.py, there are four methods at the end of the file that execute for each of the different parts of the assignment:
  - naive_bayes(stability_constant=1e-4, filename="spambase.data")
  - decision_tree(filename="spambase.data")
  - multi_class_naive_bayes(stability_constant=1e-4, filename="CTG.csv")
  - multi_class_decision_tree(filename="CTG.csv")
  
Each of the methods takes in a filename to be read in for either spambase or cardiotocography based on the method. The Naive Bayes methods also include
a stability constant parameter that will be used to stabilize our Gaussian Probability Density Function to avoid divison by zero.
  - stability_constant = 10e-4
  - filename = spambase.data or CTG.csv

How to Run:
- From the directory where this zip file was unzipped, run the following command:
python3 test_classification.py

Miscellaneous Notes:
- Operating System: Mac
- Architecture: 32 bit
- Virtual Python Environment: No