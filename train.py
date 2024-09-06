import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
# Set random seed
seed = 42


################################
########## DATA PREP ###########
################################

# Load in the data
df = pd.read_csv("train_clean.csv")

# Split into train and test sections
y = df.pop("SalePrice")
X_train, X_test, y_train, y_test = train_test_split (df, y, test_size=0.2, random_state=seed)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#################################
########## MODELLING ############
#################################

# Fit a model on the train section
model = LinearRegression()
model.fit(X_train, y_train)

# Report training set score
train_score = model.score(X_train, y_train) * 100
# Report test set score
test_score = model.score(X_test, y_test) * 100

# Write scores to a file
with open("metrics.txt", 'w') as outfile:
        outfile.write("Training variance explained: %2.1f%%\n" % train_score)
        outfile.write("Test variance explained: %2.1f%%\n" % test_score)


##########################################
##### PLOT FEATURE IMPORTANCE ############
##########################################

# For linear regression, use the model coefficients as feature importance
importances = np.abs(model.coef_)  # Use absolute value of coefficients

labels = df.columns[:-1]  

# Create a DataFrame for feature names and their importance
feature_df = pd.DataFrame(list(zip(labels, importances)), columns=["feature", "importance"])
feature_df = feature_df.sort_values(by='importance', ascending=False)

# Set the figure size before plotting (e.g., width=10, height=6)
plt.figure(figsize=(10, 8))  

# Image formatting for the plot
axis_fs = 18  # fontsize for axis labels
title_fs = 22  # fontsize for the title
sns.set(style="whitegrid")

# Create bar plot for feature importance
ax = sns.barplot(x="importance", y="feature", data=feature_df)
ax.set_xlabel('Importance', fontsize=axis_fs)
ax.set_ylabel('Feature', fontsize=axis_fs)
ax.set_title('Linear Regression\nFeature Importance', fontsize=title_fs)

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120)
plt.close()


##########################################
############ PLOT RESIDUALS  #############
##########################################

# Predict the test set
y_pred = model.predict(X_test) 

# Create DataFrame for true vs predicted values
res_df = pd.DataFrame(list(zip(y_test, y_pred)), columns=["true", "pred"])

# Plot true vs predicted values
ax = sns.scatterplot(x="true", y="pred", data=res_df)
ax.set_aspect('equal')

# Set labels and title
ax.set_xlabel('True Values', fontsize=axis_fs)
ax.set_ylabel('Predicted Values', fontsize=axis_fs)
ax.set_title('True vs Predicted Values', fontsize=title_fs)

# Add a line to show the ideal match between true and predicted
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'black', linewidth=1)

# Adjust limits based on data
plt.ylim((min(y_test), max(y_test)))
plt.xlim((min(y_test), max(y_test)))

plt.tight_layout()
plt.savefig("residuals.png", dpi=120)
plt.close()

