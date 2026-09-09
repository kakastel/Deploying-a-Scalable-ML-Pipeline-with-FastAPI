# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model was created for a Udacity Project. It uses RandomForestClassifier from scikit-learn 1.5.1. The max_depth=20 was made so that the pickle file wouldn't be as big and so the RandomForest could not just memorize the data. Throughout the code random_state=42 was used for reproducibility.

## Intended Use
This model is to be only used as a student project. It is not a model that should be deployed or used in any way to make decisions in a professional work space.

## Training Data
The census income data from 1994 consisting of 32,561 rows was split using a 75/25 split for training vs test data. Stratification was not used meaning there is not the same ratio of those who made under or over 50 thousand a year in each test and training set. Though the difference could have been large the training set had 24.2% and test set 23.7% of people earn over 50K, so not much of a difference.

The data had 8 categorical columns that were one-hot encoded into binary labels having columns go from 14 to 108. Of those 108, 102 are binary and 6 numeric.

## Evaluation Data
The evaluation data set aside from the split was never used to train any data. The total row count of the evaluation data is 8,141. This data also used the one-hot encoder and labeler that was fitted on the training test split.

## Metrics
The overall metrics of the models predictions of those making over $50K are:

- Precision: 0.7839
- Recall: 0.6098
- F1: 0.6859

The model is correct 78% of the time when predicting someone earns over $50K and properly identifies (recall) 61% of the people who made over $50K. Accuracy was not used as over 75% of people made under $50K so a model that assumed everyone made under that amount would be correct over 75% of the time and would not be helpful at all.

## Ethical Considerations
The model misses a lot of higher income women (only recalls 52% with n=2,673) more than it misses higher income men (recalls 63% with n=5,468). There is also a race gap, F1 of 0.6863 white (n=7,000) and F1 of 0.6429 for black (n=752). The race and sex are both features used in model not categories that were sliced after running the model. Interestingly for those with only a 7th-8th education the model never once predicted anyone earns over $50K in that slice (n=176).

## Caveats and Recommendations
There are some really small slices with less than 10 people where the model is perfect however since those groups are so small it is not very helpful. An example is the occupation:Armed-Forces slice with a perfect 1.000 score with only 3 rows. There are also some interesting categories like workclass where there is a question mark (?) category, so people who don't have a value in those fields still had the model make a decision.
Once again this model shouldn't be deployed and is old and outdated as its 1994 data not to be used for anything current as today things may look different with newer data.
