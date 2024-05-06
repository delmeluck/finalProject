# import libraries
from typing import List, Any

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.decomposition._pca import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# Step 1- Preprocessing
# load dataframe
data = pd.read_csv('/Users/delmeluck/Documents/Masters/INSE 6220/Exams/nfl_team_stats_2002-2023.csv')

# check if data was loaded
if len(data) > 0:
    print("\n Data description\n", data.describe())
    print(f"data loaded successfully, shape: {data.shape}")
else:
    print("no data was loaded")


# 1.1 Exploratory Data Analysis
# Data Profiling
# create a data profiling function
def create_data_profiling_df(dataset: pd.DataFrame) -> pd.DataFrame:
    # create an empty dataframe to gather information about each column
    data_profiling_df = pd.DataFrame(columns=["column_name", "column_type", "unique_values",
                                              "duplicate_values", "null_values", "max",
                                              "min", "range", "IQR"])

    # loop through each column to add rows to the data_profiling_df dataframe
    for column in dataset.columns:

        # create an empty dictionary to store the columns data
        column_dict = {}

        try:
            column_dict["column_name"] = [column]
            column_dict["column_type"] = [dataset[column].dtypes]
            column_dict["unique_values"] = [len(dataset[column].unique())]
            column_dict["duplicate_values"] = [
                (dataset[column].shape[0] - dataset[column].isna().sum()) - len(dataset[column].unique())]
            column_dict["null_values"] = [dataset[column].isna().sum()]
            column_dict["max"] = [dataset[column].max() if (dataset[column].dtypes != object) else "NA"]
            column_dict["min"] = [dataset[column].min() if (dataset[column].dtypes != object) else "NA"]
            column_dict["range"] = [
                dataset[column].max() - dataset[column].min() if (dataset[column].dtypes != object) else "NA"]
            column_dict["IQR"] = [
                dataset[column].quantile(.75) - dataset[column].quantile(.25) if (
                        dataset[column].dtypes != object) else "NA"]

        except:
            print(f"unable to read column: {column}, you may want to drop this column")

        # add the information from the columns dict to the final dataframe
        data_profiling_df = pd.concat([data_profiling_df, pd.DataFrame(column_dict)],
                                      ignore_index=True)

    # sort the final dataframe by unique values descending
    data_profiling_df.sort_values(by=['unique_values'],
                                  ascending=[False],
                                  inplace=True)

    # print the function is complete
    print(f"data profiling complete, dataframe contains {len(data_profiling_df)} columns")
    return data_profiling_df


# run the data profiling function and print the dataframe
data_profiling_df = create_data_profiling_df(dataset=data)
print("\n\nData Profiling\n", data_profiling_df)


# Removing Duplicates
# create function to remove duplicate rows based on subset (or not)
def remove_duplicates(dataset: pd.DataFrame,
                      df_name: str,
                      subset_columns: list,
                      keep="first") -> pd.DataFrame:
    num_duplicates = dataset.duplicated(subset=subset_columns).sum()
    print(f"dropping {num_duplicates} duplicate rows from {df_name}")
    dataset = dataset.drop_duplicates(subset=subset_columns, keep=keep)
    return dataset


data = remove_duplicates(dataset=data,
                         df_name="data",
                         subset_columns=["date", "season", "week", "possession_away", "possession_home", "away",
                                         "home"],
                         keep="first")


# Null value checks
# create a function to remove null values
def remove_null_values(dataset: pd.DataFrame,
                       columns: list,
                       how="any") -> pd.DataFrame:
    rows_before = dataset.shape[0]  # count the number of rows in the dataframe before dropping null values
    dataset = dataset.dropna(subset=columns, how=how)  # remove null values based on conditions

    rows_after = dataset.shape[0]  # count the number of rows in the dataframe after dropping null values

    print(f"Number of rows dropped: {rows_before - rows_after}")  # print the number of rows dropped

    return dataset


data = remove_null_values(dataset=data,
                          columns=['season', 'week', 'date', 'away', 'home', 'score_away', 'score_home',
                                   'first_downs_away',
                                   'first_downs_home', 'third_down_comp_away', 'third_down_att_away',
                                   'third_down_comp_home', 'third_down_att_home',
                                   'fourth_down_comp_away', 'fourth_down_att_away', 'fourth_down_comp_home',
                                   'fourth_down_att_home', 'plays_away',
                                   'plays_home', 'drives_away', 'drives_home', 'yards_away', 'yards_home',
                                   'pass_comp_away', 'pass_att_away',
                                   'pass_yards_away', 'pass_comp_home', 'pass_att_home', 'pass_yards_home',
                                   'sacks_num_away', 'sacks_yards_away', 'sacks_num_home',
                                   'sacks_yards_home', 'rush_att_away', 'rush_yards_away', 'rush_att_home',
                                   'rush_yards_home', 'pen_num_away', 'pen_yards_away',
                                   'pen_num_home', 'pen_yards_home', 'redzone_comp_away', 'redzone_att_away',
                                   'redzone_comp_home',
                                   'redzone_att_home', 'fumbles_away', 'fumbles_home', 'interceptions_away',
                                   'interceptions_home', 'def_st_td_away',
                                   'def_st_td_home', 'possession_away', 'possession_home'])

# Checking for columns with incorrect data types
non_numeric_columns = data.select_dtypes(include=['object']).columns
non_numeric_columns, data[non_numeric_columns].head()


# function to convert object columns to numeric
def convert_time_to_minutes(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return hours * 60 + minutes


# Applying the conversion to the 'possession_away' and 'possession_home' columns
data['possession_away'] = data['possession_away'].apply(convert_time_to_minutes)
data['possession_home'] = data['possession_home'].apply(convert_time_to_minutes)


# Data Distributions [Understanding the data]
# Plot histogram and identify outliers
def plot_histogram(df: pd.DataFrame, variable: str, bins=10, color='grey', edgecolor='black', figsize=(7, 2),
                   iqr_on=False):
    # set the figure size
    plt.figure(figsize=figsize)

    # plot the histogram
    plt.hist(df[variable],
             bins=bins,
             color=color,
             edgecolor=edgecolor)

    # customize the plot labels and colors
    plt.title(f'{variable} Histogram')
    plt.xlabel(f'{variable}')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.ticklabel_format(style='plain', axis='x')
    plt.grid(True)

    # define the Inter Quartile Range (iqr) and outlier bounds
    q1 = df[variable].quantile(0.25)
    q3 = df[variable].quantile(0.75)
    iqr = q3 - q1
    if iqr_on:
        lower_bound = q1
        upper_bound = q3
    else:
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

    # mark outlier bounds on histogram
    plt.axvline(lower_bound, color='blue', linestyle='dashed', linewidth=2, label='Lower Bound')
    plt.axvline(upper_bound, color='blue', linestyle='dashed', linewidth=2, label='Upper Bound')

    # Show the plot
    plt.legend()
    plt.show()

    # count outliers
    num_outliers = ((df[variable] < lower_bound) | (df[variable] > upper_bound)).sum()

    # print information about outliers
    if num_outliers > 0:
        print(f"{num_outliers} potential outliers detected in {variable} distribution")
        print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
    else:
        print(f"no potential outliers detected in {variable} distribution")
        print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")


for v in ['season', 'score_away', 'score_home',
          'first_downs_away', 'first_downs_home', 'third_down_comp_away',
          'third_down_att_away', 'third_down_comp_home', 'third_down_att_home',
          'fourth_down_comp_away', 'fourth_down_att_away',
          'fourth_down_comp_home', 'fourth_down_att_home', 'plays_away',
          'plays_home', 'drives_away', 'drives_home', 'yards_away', 'yards_home',
          'pass_comp_away', 'pass_att_away', 'pass_yards_away', 'pass_comp_home',
          'pass_att_home', 'pass_yards_home', 'sacks_num_away',
          'sacks_yards_away', 'sacks_num_home', 'sacks_yards_home',
          'rush_att_away', 'rush_yards_away', 'rush_att_home', 'rush_yards_home',
          'pen_num_away', 'pen_yards_away', 'pen_num_home', 'pen_yards_home',
          'redzone_comp_away', 'redzone_att_away', 'redzone_comp_home',
          'redzone_att_home', 'fumbles_away', 'fumbles_home',
          'interceptions_away', 'interceptions_home', 'def_st_td_away',
          'def_st_td_home', 'possession_away', 'possession_home']:
    plot_histogram(df=data, variable=v, bins=8)

# 2. Splitting dataset
# Creating the target variable: 1 if the home team wins, 0 otherwise
data['home_win'] = (data['score_home'] > data['score_away']).astype(int)

# Selecting potential features excluding identifiers and final scores
features = data.columns.difference(['season', 'week', 'date', 'away', 'home', 'score_away', 'score_home', 'home_win'])

# Splitting data into features (X) and target (y)
X = data[features]
y = data['home_win']

# Splitting the dataset into training and testing sets
# 1 - Training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 2 - Scaled Test and Train datasets
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y, test_size=0.2,
                                                                                random_state=42)

# 3 - PCA splitting
pca = PCA(n_components=0.95)  # Keeping 95% of the variance
X_pca = pca.fit_transform(X_scaled)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Explained variance ratio by PCA components
explained_variance_ratio = pca.explained_variance_ratio_

print("\nexplained_variance_ratio: ", explained_variance_ratio, "\nX_Train PCA shape", pca.fit_transform(X).shape)

# Plotting the boxplot for each numerical feature dataset
plt.figure(figsize=(20, 16))
sns.boxplot(data=features)
plt.xticks(rotation=90)
plt.title('Boxplot of NFL Data Features')
plt.show()

# Calculate eigenvalues and eigenvectors from PCA
eigenvalues = pca.explained_variance_
eigenvectors = pca.components_

print("\nEigenvalues: ", eigenvalues, "\nEigenvectors", eigenvectors, '\n')

# Feature Engineering
corr_matrix = X.corr().abs()  # create correlation matrix

upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))  # the upper triangle of correlation matrix

# plot the heatmap of the upper triangle
plt.figure(figsize=(8, 6))
sns.heatmap(upper_triangle, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Features')
plt.show()


# function to drop highly correlated features
def find_highly_correlated_features(X_data: pd.DataFrame, threshold=0.8) -> list[Any]:
    # create a  correlation matrix
    corr_matrix = X_data.corr().abs()

    # select the upper triangle of correlation matrix
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # find features with correlation greater than the threshold
    features_to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

    # print and return the features to drop
    print(f"features dropped: {features_to_drop}")
    return features_to_drop


features_to_drop = find_highly_correlated_features(X_data=X, threshold=0.8)
X_train_feature_selection = X_train.drop(columns=features_to_drop, axis=1)
X_test_feature_selection = X_test.drop(columns=features_to_drop, axis=1)
X_train_feature_selection.sample(3)


# Model building using Gradient Booster
def fit_and_evaluate_model(X_train, X_test, y_train, y_test, feature_engineering: str):
    print(f'-' * 70)
    print(f"Model with feature engineering techniques : {feature_engineering}")

    # Initialize models
    gb_classifier = GradientBoostingClassifier(random_state=42)
    gb_classifier.fit(X_train, y_train)  # Training the classifier
    y_pred_gb = gb_classifier.predict(X_test)  # Predicting the test set results

    # Predict probabilities (useful for obtaining probability estimates)
    y_pred_proba = gb_classifier.predict_proba(X_test)[:, 1]

    classification_report_result_gb = classification_report(y_test, y_pred_gb)  # Evaluate the model

    print('\n Classification report: Gradient Booster\n', classification_report_result_gb)
    print("\nPrediction Probability\n", y_pred_proba)


def fit_and_evaluate_scaled_model(scaled_X_train, scaled_X_test, scaled_y_train, scale_y_test,
                                  feature_engineering: str):
    print(f'-' * 70)
    print(f"Model with feature engineering techniques : {feature_engineering}")

    # Initialize models
    gb_classifier = GradientBoostingClassifier(random_state=42)
    linear_model = LinearRegression()
    random_forest_model = RandomForestRegressor(random_state=42)
    neural_network_model = MLPRegressor(random_state=42, max_iter=500)

    # Training the classifier
    gb_classifier.fit(scaled_X_train, scaled_y_train)
    random_forest_model.fit(scaled_X_train, scaled_y_train)
    neural_network_model.fit(scaled_X_train, scaled_y_train)
    linear_model.fit(scaled_X_train, scaled_y_train)

    # Predicting the test set results
    y_pred_gb = gb_classifier.predict(scaled_X_test)
    y_pred_rrm = random_forest_model.predict(scaled_X_test)
    y_pred_nnm = neural_network_model.predict(scaled_X_test)
    y_pred_lm = np.where(linear_model.predict(scaled_X_test) > 0.5, 1, 0)

    # Evaluate the model
    classification_report_result_gb = classification_report(scale_y_test, y_pred_gb)
    classification_report_result_lm = classification_report(scale_y_test, y_pred_lm)

    print('\n Classification report: Gradient Booster\n', classification_report_result_gb)
    print('\n Classification report: Linear Regression\n', classification_report_result_lm)

    # Predict and evaluate each model
    models = [linear_model, gb_classifier, random_forest_model, neural_network_model]
    model_names = ["Linear Regression", "Gradient Boosting", "Random Forest", "Neural Network"]
    results = []

    for model, name in zip(models, model_names):
        predictions = model.predict(scaled_X_test)
        mse = mean_squared_error(scale_y_test, predictions)
        r2 = r2_score(scale_y_test, predictions)
        results.append((name, mse, r2,))

    print("\nResults:\nName\t\t\t\tMean Squared Error\t\t\tR2")
    for result in results:
        print(result, '\n')


def fit_and_evaluate_pca_model(pca_X_train, pca_X_test, pca_y_train, pca_y_test, feature_engineering: str):
    print(f'-' * 70)
    print(f"Model with feature engineering techniques : {feature_engineering}")

    # Initialize models
    gb_classifier = GradientBoostingClassifier(random_state=42)
    linear_model = LinearRegression()
    random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
    neural_network_model = MLPRegressor(hidden_layer_sizes=(50,), max_iter=500)

    # Training the classifier
    gb_classifier.fit(pca_X_train, pca_y_train)
    random_forest_model.fit(pca_X_train, pca_y_train)
    neural_network_model.fit(pca_X_train, pca_y_train)
    linear_model.fit(pca_X_train, pca_y_train)

    # Predict probabilities (useful for obtaining probability estimates)
    y_pred_proba = gb_classifier.predict_proba(pca_X_test)[:, 1]

    # Predicting the test set results
    y_pred_gb = gb_classifier.predict(pca_X_test)
    y_pred_rrm = random_forest_model.predict(pca_X_test)
    y_pred_nnm = neural_network_model.predict(pca_X_test)
    y_pred_lm = np.where(linear_model.predict(pca_X_test) > 0.5, 1, 0)

    # Evaluate the model
    classification_report_result_gb = classification_report(pca_y_test, y_pred_gb)
    classification_report_result_lm = classification_report(pca_y_test, y_pred_lm)

    print('\n Classification report: Gradient Booster\n', classification_report_result_gb)
    print('\n Classification report: Linear Regression\n', classification_report_result_lm)

    # Predict and evaluate each model
    models = [linear_model, gb_classifier, random_forest_model, neural_network_model]
    model_names = ["Linear Regression", "Gradient Boosting", "Random Forest", "Neural Network"]
    results = []

    for model, name in zip(models, model_names):
        predictions = model.predict(pca_X_test)
        mse = mean_squared_error(pca_y_test, predictions)
        r2 = r2_score(pca_y_test, predictions)
        results.append((name, mse, r2))

    print("\nResults:\nName\t\t\t\tMean Squared Error\t\t\tR2")
    for result in results:
        print(result)


print("\nNo scaling & PCA")
fit_and_evaluate_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                       feature_engineering="Feature_selection")

print("\n Standardized Data")
fit_and_evaluate_scaled_model(scaled_X_train=X_train_scaled, scaled_X_test=X_test_scaled, scaled_y_train=y_train_scaled,
                              scale_y_test=y_test_scaled, feature_engineering="Standardized")

print("\n PCA Data")
fit_and_evaluate_pca_model(pca_X_train=X_train_pca, pca_X_test=X_test_pca, pca_y_train=y_train_pca,
                           pca_y_test=y_test_pca, feature_engineering="PCA")
