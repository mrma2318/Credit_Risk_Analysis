# Credit_Risk_Analysis
## Overview: Use data preparation, statisical reasoning, and machine learning to solve credit card risk. 

### The purpose of this analysis is to use LendingClub's credit card dataset and employe different techniques to train and evaluate models with unbalanced classes. The three technical analysis I will use is the Resampling Models, the SMOTEENN Algorithm, and Ensemble Classifiers to predict credit risk. 

## Analysis
### Resampling Models to Predict Credit Risk
- When I uploaded the data and read it into a DataFrame, I needed to create the training variables by converting the string values into numberical ones using the get_dummies() method. Then I needed to create my target variables and balance them before I could split the data into training and testing data. 

- Now I can resample the data by using the oversampling algorithms (naive random oversampling and SMOTE) to determine which algorithm results in the best performance. Then, I used the undersampling algorithm cluster centroids to resample the data to also determine which algorithm results in the best performance compared to the oversampling algorithms. 

- For each resampling algorithm first needed to resample the data using the correct algorithm type (for example, SMOTE for oversampling). Then I used the Logicistic Regression model using the resampled data to make predicitions and evaluate the model's performance. I also calculated the accuracy score, generated a confusion matrix, and printed out the imbalanced classification report. See Image 1 of the undersampling algorithm. 

### Image 1: Undersampling Algorithm
![Undersampling Algorithm](https://github.com/mrma2318/Credit_Risk_Analysis/blob/4c91ebadcdafa788a68adfc572d3abaa9d50da06/images/undersampling.png)

### SMOTEENN Algorithm to Predict Credit Risk
- In addition to oversampling and undersampling algorithms, there is also combination (over and under) sampling algorihtm, SMOTEENN. For the SMOTEENN algorithm, I used a combinatorial approach of the over and undersampling to determine if the results from the combinatorial approach are better at predicting credit risk than the resampling algorithms. 

- After the data was resampled using SMOTEENN, I used the Logistic Regression classifier again to make predictions and evaluate the model's performance. Then I calculated the accuracy score of the model, generated a confusion matrix, and printed out the imbalanced classification report, Image 2.

### Image 2: SMOTEENN Algorithm
![SMOTEENN Algorithm](https://github.com/mrma2318/Credit_Risk_Analysis/blob/4c91ebadcdafa788a68adfc572d3abaa9d50da06/images/SMOTEENN.png)

### Ensemble Classifiers to Predict Credit Risk
- Lastly, I compared two ensemble algorithms (Balanced Random Forest Classifer and Easy Ensemble Classifer), to determine which algorithm results in the best performance. Once again, after uploading the data and reading it into a DataFrame, I needed to create the training variables. I did this by converting the string values into numberical ones using the get_dummies() method. Then I needed to create my target variables and balance them before I could split the data into training and testing data. 

- For the ensemble classifiers, I used the balanced random forest classifer first. I resampled the training data by using the BalancedRandomForestClassifier algorithm with 100 estimators. Then I calculated the accuracy score of the model, generated the confusion matrix, and printed out the imbalanced classification report, Image 3. I also printed the feature importance, sorted in descending order, with the feature score. 

### Image 3: Balanced Random Forest Classifer
![Balanced Random Forest Classifer](https://github.com/mrma2318/Credit_Risk_Analysis/blob/4c91ebadcdafa788a68adfc572d3abaa9d50da06/images/balanced_classifier.png)

- Then I resampled the training data using the EasyEnsembleClassifier algorithm with 100 estimators as well. Just like for the BalancedRandomForestClassifier algorithm, I calculated the accuracy score of the model, generated the confusion matrix, and printed out the imbalanced classification report, Image 4.

### Image 4: Easy Ensemble AdaBoost Classifer
![Easy Ensemble AdaBoost Classifer](https://github.com/mrma2318/Credit_Risk_Analysis/blob/4c91ebadcdafa788a68adfc572d3abaa9d50da06/images/adaboost_classifier.png)

## Results: Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all six machine learning models. Use screenshots of your outputs to support your results.
- When reviewing the balanced accuracy scores, I want a score that is closer to 1 than 0. The balanced accuracy score provides the predicition of the models performance. Typically, an accuracy score greater than 70% is ideal. For the Naive Random Oversampling, the balanced accuracy score is 64%, Image 5. The SMOTE balanced accuracy score is also below 70% with a score of 66%. Since both of these are fairly close to 70%, these models  might work, but we will know more after reviewing the precision and recall scores as well. However, the undersampling and combination balanced accuracy scores are both at 54%, therefore, these algorithms might not be the best at predicting the models performance, Image 1 and Image 6. 

### Image 5: Naive Balanced Accuracy Score
![Naive Balanced Accuracy Score](https://github.com/mrma2318/Credit_Risk_Analysis/blob/4c91ebadcdafa788a68adfc572d3abaa9d50da06/images/naive_balanced_score.png)

### Image 6: Combination Balanced Accuracy Score
![Combination Balanced Accuracy Score](https://github.com/mrma2318/Credit_Risk_Analysis/blob/4c91ebadcdafa788a68adfc572d3abaa9d50da06/images/comb_balanced_score.png)

- When looking at the ensebmle learners however, they both express high balanced accuracy scores. For the balanced random forest classifer, it has a balanced accuracy score of 79%, indicating this algorithm would be good at predicting the models performance, Image 7. In addition, the Easy Ensemble AdaBoost Classifier has a balanced accuracy score of 93%, Image 8, indicating that out of all the algorithms, this is best algorithm at predicting the models performance. However, before I can make that determination, I still need to review the precision and recall scores. 

### Image 7: Balanced Random Forest Balanced Accuracy Score
![Balanced Random Forest Balanced Accuracy Score](https://github.com/mrma2318/Credit_Risk_Analysis/blob/4c91ebadcdafa788a68adfc572d3abaa9d50da06/images/forest_balanced_score.png)

### Image 8: Easy Ensemble AdaBoost Balanced Accuracy Score
![Easy Ensemble AdaBoost Balanced Accuracy Score](https://github.com/mrma2318/Credit_Risk_Analysis/blob/4c91ebadcdafa788a68adfc572d3abaa9d50da06/images/adaboost_balanced_score.png)

- The precision and recall scores let us know out of all the predicitions, which ones are positive. Therefore, the precision score is providing the percentage of results that are truley positive, whereas the recall score provides the percentage of those that are predicted positive. For this analysis, we don't want to miss any high credit risks because we don't want to give someone credit that's a high risk. However, we also don't want to miss anyone who is low credit risk. 

- First, looking at the high risk for each algorithm, I'll want a low precision score and a high recall score. This is because we want to make sure that we are identifying all the high credit risks. For all the different algorithms, the percentage ranged between 1% and 9%. This indicates that there are very few high credit risk being marked incorrectly as low credit risk. 

- To know what a good recall score is, I'll use the similar precentage range as I did for the balanced accuracy score, with a score greater than 70% being a good percentage. Looking at the resampling algorithms, naive random oversampling had a recall of 70%, and SMOTEENN had a recall of 73%. This indicates these two algorithms had a high percentage of predicting high credit risks. However, SMOTE oversampling had a recall of 63%, and Cluster Centroid undersampling had a recall of 69%. Since these two algorithms are close to 70% they predict true high risk credit risks, but not as effecient as the others. 

- When reviewing the ensemble learner, the balanced random forest classifer has a recall of 70%. This indicates that this model is also good at predicitng high risk credit risks. However, the Easy Ensemble AdaBoost Classifer has a recall of 92%. Compared to all the other algorithms, the Easy Ensemble AdaBoost Classifer predicts high risk credit risks, 92% of time.

- Next, looking at the low risk for each algorithm, ideally I would want high precision and high recall. This is because we want to make sure that we don't miss any low credit risks, so we want all low risk results labeled correctly. For all the algorithms, they have a precision of 100%, indiciating all the models predict true low credit risks 100% of the time. However, the recall scores for all the resampling algorithms were less than 70%, indicating they may not be the best algorithms for predicting true low credit risks. Naive oversampling had a recall of 59%, SMOTE had a recall score of 69%, cluster centroids had a recall of 40%, and the SMOTEENN algorithm had a recall of 57%. Out of the resampling algorithms, SMOTE is the closest to 70% indicaing it might be a good model in predicting low credit risk.

- Similar to the resampling algorithms, the ensemble learners also had 100% precision for low credit risk. The balanced random forest classifer had a recall of 87%, indicating this model does a great job in providing results that are truly low credit risk. The Easy Ensemble AdaBoost Classifier also has a high recall score of 94%, indicating this algorithm also does a great job in providing results that are truly low credit risk. 

## Summary: Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.
- In conclusion, due to all the recall scores for the resampling models, I wouldn't recommend using either of them to use. We want to make sure we correctly identify those that are high credit risk as well as those that are low credit risk. We don't want to give someone who is high risk credit, but also don't want to deny someone that is low risk because they were incorrectly categorized as high risk. Therefore, I would suggest using the Easy Ensemble AdaBoost Classifier. Even though this model has a higher high risk precision percentage at 9%, it also has high percentages in recall and precision for low risk. The recall for high credit risk is 92%, indicating that 92% of the time, it correctly identifies true high credit risk. The Easy Ensemble AdaBoost Classifier almost 100% of the time provides all the true low credit risks with a precision of 100% and a recall of 94%. 