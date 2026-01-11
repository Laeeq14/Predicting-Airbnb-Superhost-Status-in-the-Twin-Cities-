Report Link: [group23_final_report.pdf](https://github.com/user-attachments/files/24547128/group23_final_report.pdf)

PROBLEM STATEMENT: <br/>
> Superhosts are trusted Airbnb hosts who provide high-quality stays.
> Predicting which hosts are superhosts on Airbnb is difficult due to the size and complexity of the dataset.
> Lots of key factors : reviews, pricing, amenities, reliability.
> Can we predict superhosts using a machine learning model?

PROPOSED SOLUTION:
> Performing data preprocessing and feature engineering on datasets.
> Analyzing patterns using EDA (Exploratory Data Analysis) 
> Building and evaluating various classification models.
> Identifying the most important features.
> Identifying the classification model with the best performance.

IMPORTANCE OF IDENTIFYING SUPERHOSTS
> Improves evaluation: It provides a reliable way for guests to quickly evaluate potential hosts and for hosts to measure their performance against top performers.<BR/>
> Increases transparency: The Superhost badge signifies a host's commitment to quality, helping guests make informed booking decisions.<BR/>
> Boosts visibility: Listings with the Superhost badge can appear higher in search results, leading to more bookings for the host.<BR/>

DATASET OVERVIEW
> Source: Airbnb open dataset (https://insideairbnb.com/get-the-data/)<BR/>
Datasets used: <BR/>
listings.csv - host, listing, location, amenities, ratings<BR/>
reviews.csv - review text + timestamps<BR/>
Number of observations: 5000+ observations<BR/>
No. of features (before preprocessing): 80+<BR/>
Target variable: host_is_superhost<BR/>
Class Distribution:<BR/>
1 (Superhost): 53.7%<BR/>
0 (Not Superhost): 46.3%<BR/>

DATA PREPROCESSING
> Removed duplicate listings (id) <BR/>
Converted host_since to datetime<BR/>
Converted percentage features to numeric values<BR/>
Extracted numeric values from bathrooms_text using regex<BR/>
Handled missing values:<BR/>
Numeric - median imputation<BR/>
Categorical - “missing” placeholder<BR/><BR/>

> Preprocessing Pipeline (Sklearn)<BR/>
Used ColumnTransformer + Pipeline<BR/>
One-Hot Encoding for categorical features<BR/>
StandardScaler for numerical features<BR/>

FEATURE ENGINEERING
> New features created:<BR/>
host_experience_years: How long host has been active<BR/>
num_amenities: Total amenities offered<BR/>
review_count: Number of reviews per listing<BR/>
avg_comment_length: Average length of comments<BR/>
reviews_per_month: Hosting activity<BR/>
log1p(skewed_features): Reduce skewness<BR/>

> Merging Review data<BR/>
Joined reviews.csv with listings.csv using listing_id = id<BR/>
Aggregated:<BR/>
Review count<BR/>
Average text length<BR/>

EDA
> <img width="202" height="91" alt="image" src="https://github.com/user-attachments/assets/ff59b521-341e-4a89-a8a5-8f93b29cf61c" /> Dataset is balanced enough for classification
> <img width="847" height="528" alt="image" src="https://github.com/user-attachments/assets/f5d107cd-82a4-4cd6-b88c-11f5f35ddd9b" />
> <img width="852" height="529" alt="image" src="https://github.com/user-attachments/assets/9aa0162a-252c-41b0-b137-f4fbd4fe6ab9" />
> <img width="648" height="396" alt="image" src="https://github.com/user-attachments/assets/d3236427-53b1-4b8f-b596-715721d41d96" />
> <img width="626" height="396" alt="image" src="https://github.com/user-attachments/assets/8926da19-b186-448e-b4ef-977933d9743f" />
> <img width="634" height="396" alt="image" src="https://github.com/user-attachments/assets/52fcddf6-5739-4972-9035-bb498e526338" /><br/>
> Superhosts generally have:
Higher review scores
More amenities
More reviews per month
Longer hosting experience
Applied log transformation to skewed features
Visualizations confirmed distinct patterns between classes - classification is feasible.

CORRELATION INSIGHTS
> <img width="983" height="878" alt="image" src="https://github.com/user-attachments/assets/80cb6808-d953-4284-ac8f-d7f2e183aa15" />
> <img width="341" height="295" alt="Screenshot 2026-01-10 180020" src="https://github.com/user-attachments/assets/d905b502-5797-4f92-be1b-85750001412a" />

MODEL SELECTION
> Methodology
train_test_split() with stratified sampling
Evaluation metrics: accuracy, F1, ROC-AUC
Hyperparameter tuning using RandomizedSearchCV + CV
Used pipelines to combine preprocessing + model training
> <img width="636" height="324" alt="image" src="https://github.com/user-attachments/assets/6d8c75d3-1798-4f2c-83e1-08dc1f54fe52" />

MODEL PERFORMANCE
> Logistic Regression <br/>
> <img width="801" height="879" alt="image" src="https://github.com/user-attachments/assets/df77bea9-f92b-4d4d-98f3-704bab8180e8" /> <br/><br/>
> Decision Tree <br/>
> <img width="794" height="816" alt="image" src="https://github.com/user-attachments/assets/6e33a531-f9fb-4f2a-980e-cfcc57fa698a" /> <br/><br/>
> Random Forest <br/>
> <img width="792" height="825" alt="image" src="https://github.com/user-attachments/assets/436eff48-3f5e-4b66-ab7d-54831fd3977a" /> <br/><br/>
> XGBoost <br/>
> <img width="790" height="825" alt="image" src="https://github.com/user-attachments/assets/7941be10-0e85-4bc4-bc19-8e598bd47d6f" />

HYPERPARAMETER TUNING
> Objective: Improve model performance using RandomizedSearchCV with StratifiedKFold (5-fold) validation.
Why Hyperparameter Tuning?
Prevents overfitting / underfitting
Finds optimal model settings
Improves generalization on unseen data
Method used: RandomizedSearchCV + StratifiedKFold(cv=5, scoring="f1")
Tuned Parameters:
Random Forest: n_estimators, max_depth, max_features, min_samples_split
XGBoost: n_estimators, max_depth, learning_rate, subsample, colsample_bytree <br/><br/>
> Random Forest (Tuned) <br/>
> <img width="885" height="690" alt="image" src="https://github.com/user-attachments/assets/c6d0a5d9-3002-4bcc-a9c1-3cd11b61f905" /> <br/><br/>
> XGBoost (Tuned) <br/>
> <img width="1050" height="687" alt="image" src="https://github.com/user-attachments/assets/53927e7a-90da-4ce4-bd41-b71060470952" /> <br/><br/>
> <img width="928" height="155" alt="image" src="https://github.com/user-attachments/assets/2657526b-f550-469d-a781-95bbcea91825" />

ROC CURVE AND PRECISION RECALL CURVES
> <img width="1183" height="484" alt="image" src="https://github.com/user-attachments/assets/52c9a626-5057-42b1-90b2-9acbe01ace42" /> <br/>

BEST MODEL - XGBOOST (TUNED)
> Reasons<br/>
> 1. Highest overall performance: XGBoost consistently
outperforms all other models across key evaluation
metrics<br/>
> 2. Handles Class Imbalance Effectively: XGBoost uses
scale_pos_weight and gradient boosting, which
improve recall for minority class, helps detect
Superhosts more reliably<br/>
> 3. Captures Complex Feature Interactions: XGBoost
uses tree boosting, allowing it to model non-linear
relationships, able to learn feature interactions that
traditional logistic regression & decision trees miss<br/>
> 4. Faster & More Efficient: Built-in regularization (L1 &
L2) reduces overfitting<br/>

FEATURE IMPORTANCE
> <img width="882" height="550" alt="image" src="https://github.com/user-attachments/assets/855820f6-2b2a-4055-aa0e-0036786f559a" /> <br/>
> Random Forest Feature Importance: <br/>
> review_scores_rating, host_experience_years, and
reviews_per_month are the strongest signals for the
random forest model when determining superhost
status. <br/><br/>
> <img width="1004" height="550" alt="image" src="https://github.com/user-attachments/assets/b1f43613-da6f-4c05-b8e5-262efe61f281" />
> Top Logistic Regression Coefficients: <br/>
> The models suggest that while a high review_scores_rating is
essential for superhost status, having a very large number of
listings (host_listings_count being highly negative) works
against achieving that status, perhaps due to difficulty
maintaining high standards across many properties.

SHAP ANALYSIS
> What is SHAP Analysis?<BR/>
SHAP stands for SHapley Additive exPlanations.<BR/>
It is a powerful framework based on cooperative game theory
that explains the output of any machine learning model.<BR/>
It assigns an "importance value" (a SHAP value) to each feature for a specific prediction, showing how much that feature pushed
the prediction from the average baseline. <BR/><BR/>
> Why is it Used?<BR/>
Interpretability: Provides clear, human-understandable reasons
for complex model decisions (e.g., why a host was predicted to
be a superhost).<BR/>
Fairness & Debugging: Helps data scientists identify hidden
biases and ensure models behave as expected.<BR/>
Feature Understanding: Visualizes global and local feature
relationships, including complex interactions between features.<BR/>

SHAP ANALYSIS RESULTS
> <img width="492" height="680" alt="image" src="https://github.com/user-attachments/assets/4df26f1e-72b3-415c-a5bb-4df4eac44334" /><BR/>
> The image visualizes
the interaction effects between a host's
acceptance rate and response rate when predicting superhost
status.<BR/><BR/>
Interaction:
The model uses these two features together; they
have little impact in isolation.<BR/>
Synergy:
High values for both rates (red dots on the right side)
create a strong positive effect on the prediction.<BR/>
Detriment:
Low values for both rates (blue dots on the left side)
create a negative effect on the prediction.<BR/>
Conclusion:
A host needs both high acceptance and high
response rates to benefit the superhost prediction.<BR/>

