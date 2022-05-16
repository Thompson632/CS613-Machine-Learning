CS-613 Machine Learning
Assignment 4 - Dimensionality Reduction
Robert Thompson

Versions:
- Python: 3.9.9
- pip: 22.0.4
- numpy: 1.22.3

Description:
- In test_dimensionality_reduction.py, there are four methods at the end of the file that execute for each of the different parts of the assignment:
  - pca(filename="lfw20.csv", num_components=2)
  - knn(filename="lfw20.csv", k=1)
  - knn_pca(filename="lfw20.csv", k=1, num_components=100)
  - eigenfaces_compression(filename="lfw20.csv", num_components=1, person_index=224)

Each of the methods take in a filename to be read in and other useful parameters described below:
  - filename = lfw20.csv
    - The filename to be read in
  - num_components = 100
    - Number of Components for our PCA Model
  - k = 1
    - Number of K-Neighbors for our KNN Classifier
  - person_index = 224
    - This is the index of Tiger Woods in our dataset

How to Run:
- From the directory where this zip file was unzipped, run the following command:
python3 test_dimensionality_reduction.py

Miscellaneous Notes:
- Operating System: Mac
- Architecture: 32 bit
- Virtual Python Environment: No