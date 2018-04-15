### Setup

1)  In your project root folder, create a folder called 'dataset'
2)  Download this dataset: https://www.kaggle.com/danofer/movies-data-clean/output into the 'dataset' folder
3)  Run data_prep.ipynb to generate our version of the dataset
4) Add SYDE-522-Project/ directory to your $PYTHONPATH
5) Make sure you are running Python 2.7
Justin


### How To Run

1) Locate run.py under movie_rating_classifiation
2) Run using: python run.py `<classifier>` (svm, rfc, gbc, mlp, lr)

Options:

-p
To run data preprocessing before classifying

-o
To run optimization instead of just classifying

-t
To tune classifier parameters

-h
Help
