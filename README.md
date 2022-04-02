# Overview of the Credit_Risk_Analysis
The purpose of this challenge was to use machine learning to predict credit card risk. We tested multiple algorithms for 
accuracy using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, to see 
which one best forecasts low and high-risk loan applications.

Each loan has about 85 features in the dataset. A couple of examples are Principal and Interest 
Received to Date, Most Recent Payment Amount, Interest Rate, Debt-to-Income Ratio, Months Since recent Credit Inquiry, and Home Ownership.

The issue with this dataset is that it has a significant bias in favour of good loans. Because most loans never default,
 99.9% of the loans in the database are considered low-risk. That's a lot of skewed data.

We utilize sklearn to split the data into training and testing sets to overcome the skewness of the data.
The testing data is then used to train models and make predictions.

The following criteria are used to assess the model's performance:

- Accuracy Score - This is just the percentage of right predictions, with 1 being 100% accurate and 0 being 0% accurate.

- Precision Score - a metric for how reliable a positive classification is, with 1 being 100% and 0 being 0%. As an example, "I'm aware that the high-risk test was positive. How  likely is it that the loan will be high-risk?"

- Recall Score - a measure of how many actual positives were accurately detected, with 1 being 100% correct and 0 being 0% correct. "I'm well aware that my loan is a high-risk investment. How likely is  that the test will be able to predict it?"

## Results

We will look at six different machine learning models that can predict a high-risk loan application.

### Random Oversampling

- Random oversampling randomly selects instances of minority classes and adds them to the training set until the majority and minority classes are balanced.

- Accuracy: Expected to be 0.639 for high-risk use, which is actually correct.

![random oversampling](https://user-images.githubusercontent.com/92459399/161367071-9d953a34-cb5e-4ff7-82e2-b1494c447025.PNG)

**Precision:** A high-risk application of 0.01 is predicted and is actually correct. 

**Recall:** 0.61 of the actual high-risk application was correctly identified. 

![Random Oversampling Imbalanced](https://user-images.githubusercontent.com/92459399/161367165-7701bddd-939d-4a76-ad15-8d41cc1773f8.PNG)


### SMOTE Oversampling

- The synthetic Minority Oversampling (SMOTE) technique increases the size of minorities by interpolating new instances. That is, some nearest neighbours are selected for an instance of the minority class. 

- Accuracy: Expected to be 0.625 for high-risk use, which is actually correct. 

![Smote Oversampling balanced accuracy](https://user-images.githubusercontent.com/92459399/161367209-9f32dcd3-a156-45be-b60d-fd4a453c9521.PNG)

**Precision:** A high-risk application of 0.01 is predicted and is actually correct.

**Recall:** 0.62 of the actual high-risk application was correctly identified. 
 
![Smote Oversampling imbalanced report](https://user-images.githubusercontent.com/92459399/161367224-8b6c94e4-2f5b-4610-9006-4d71eeea2953.PNG)

### Cluster Centroids Undersampling

- Cluster Centroids identify majority class clusters and generate synthetic data points called centroids that represent the clusters. The majority class is then subsampled to the size of the minority class. 

- Accuracy: 0.515 for high-risk applications was predicted and was actually correct. 

![Cluster Centroids Undersampling](https://user-images.githubusercontent.com/92459399/161367244-d445f856-f902-4b41-a3f8-e3c73eb63317.PNG)

**Precision:** A high-risk application of 0.01 is predicted and is actually correct.

**Recall:** 0.60 of the actual high-risk application was correctly identified. 

![Cluster Centroids Imbalanced Classification](https://user-images.githubusercontent.com/92459399/161367257-51bc8bc5-67a9-425c-bac3-9b7399132cde.PNG)

### SMOTEENN Combination Sampling

- The SMOTEENN is a combination of the SMOTE algorithm and the Edited Nearest  Neighbors (ENN) algorithm. SMOTEENN is a two-step process. 

*1.* Oversample  minority classes using SMOTE.

*2.* Use an undersampling strategy to clean up the resulting data. If the two nearest neighbors of a data point belong to two different classes, the data point will be deleted. 

- Accuracy: Expected high risk use of 0.625, actually correct.

![Smote Oversampling balanced accuracy](https://user-images.githubusercontent.com/92459399/161367307-ea04c079-edcb-46a4-8914-94f3c97ef3bb.PNG)

**Precision:** A high-risk application of 0.01 is predicted and is actually correct.

**Recall:** 0.70 of the actual high-risk application was correctly identified. 

![Smoteen Imbalanced Classification](https://user-images.githubusercontent.com/92459399/161367336-a1240bb2-fd52-4d5d-a283-78ee20e63dbc.PNG)

### Random Forest Classifier

- The random forest algorithm will sample the data and build several smaller, simpler decision trees. Each tree is simpler because it is built from a random subset of features.

- Accuracy: 0.788 of high risk applications were predicted and actually correct.

![Balanced Random Forest](https://user-images.githubusercontent.com/92459399/161367412-a215c62f-3ac5-4ea7-be0b-b7fbd088091e.PNG)

**Precision:** A high-risk application of 0.03 is predicted and is actually correct.

**Recall:** 0.70 of the actual high-risk application was correctly identified. 

![Balanced Random Forest Imbalanced Report](https://user-images.githubusercontent.com/92459399/161367440-6f6f5dae-354d-497d-8a92-49d49bddbf69.PNG)

### Easy Ensemble Classifier

- Easy Ensemble selects all examples from the minority class and a subset from the majority class to create a balanced sample of the training set. Instead of using a pruned decision tree, a boosted decision tree is used for each subset, especially the AdaBoost algorithm.  

- Accuracy: 0.788 for high-risk use is expected and is actually correct.

![Easy Ensemble Classifier](https://user-images.githubusercontent.com/92459399/161367712-a8c560e2-0025-41f8-8965-4805107e7566.PNG)

**Precision:** A high-risk application of 0.03 is predicted and is actually correct.

**Recall:** 0.70 of the actual high-risk application was correctly identified.

![Easy Ensembe Imbalanced Report](https://user-images.githubusercontent.com/92459399/161367721-94bbde69-3b36-46fa-a294-8cb1b10a77b1.PNG)

## Summary

It is interesting to find out that some of the above machine learning models outperform others. Given that, further study is needed to identify machine learning models that are more successful at making predictions.

However, considering the multiple methods listed above, the Easy Ensemble model is the one which I would recommend for this scenario because each of its scores reveals that it is most likely to identify and anticipate high-risk loan applications effectively.
