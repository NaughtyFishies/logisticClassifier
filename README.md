# logisticClassifier
This code creates a logistic regression classification to predict a the outcome of binarily classified data. More specifically, for this code we used loan applicant data to make a prediction about the applicant's loan status.

### Data
The data used for this project can be found [here](https://www.kaggle.com/datasets/parthpatel2130/realistic-loan-approval-dataset-us-and-canada).

## How to use
### Required libraries
This code uses numpy, pandas, scikit-learn and matplotlib (for graphing loss) and runs on python 3.
These can be installed with
```bash
pip install numpy pandas scikit-learn matplotlib
```
### Using the model
It's assumed that the data for this model is in csv format and located in a subdirectory from the python module name "data".
To run the model, type in the following into the terminal:
```bash
python3 classifier.py
```