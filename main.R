# imports
install.packages(c("tidyverse", "caret", "randomForest", "xgboost", "janitor"))

library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(janitor)

# loading and preparing the data
loan_data <- read.csv("Loan_default.csv") %>%
  clean_names() # This line fixes the column name issue

# Removing the LoanID column, dont need it
# Convert the 'Default' column to a factor
loan_data <- loan_data %>% 
  select(-loan_id) %>%
  mutate(default = as.factor(default))

# Identifying which columns are categorical (with a small number of unique values)
# and which are numerical.
categorical_cols <- sapply(loan_data, function(x) length(unique(x)))
one_hot_cols <- names(categorical_cols[categorical_cols < 10 & names(categorical_cols) != "default"])
numerical_cols <- names(categorical_cols[categorical_cols >= 10 & names(categorical_cols) != "default"])

# Applying one-hot encoding only to the selected categorical columns
# The 'sep' argument ensures that column names are valid R names,
# which fixes the 'EducationHigh School' error.
dummy_variables <- dummyVars(~ ., data = loan_data %>% select(one_hot_cols), fullRank = TRUE)
one_hot_data <- predict(dummy_variables, newdata = loan_data)

# Combining the one-hot encoded data with the original numerical columns and the target variable
# This ensures a complete and consistent dataset for the models.
final_data <- cbind(loan_data %>% select(all_of(numerical_cols), default), one_hot_data)

glimpse(final_data)

# splitting the data for training and testing
# Split the data into an 80/20 training and testing set
set.seed(123)
train_index <- createDataPartition(final_data$default, p = 0.8, list = FALSE)
training_set <- final_data[train_index, ]
testing_set <- final_data[-train_index, ]

# Trainging and comparing models
# Training the Random Forest model
set.seed(123)
rf_model <- randomForest(default ~ ., data = training_set, ntree = 500)

# Make predictions on the testing set
rf_predictions <- predict(rf_model, testing_set)

# Evaluate the model's performance
rf_cm <- confusionMatrix(rf_predictions, testing_set$default)
print("Random Forest Confusion Matrix:")
print(rf_cm)

# XGBoost model
# Prepare data for XGBoost (convert to a matrix format)
train_matrix <- xgb.DMatrix(data = as.matrix(training_set %>% select(-default)), label = as.integer(training_set$default) - 1)
test_matrix <- xgb.DMatrix(data = as.matrix(testing_set %>% select(-default)), label = as.integer(testing_set$default) - 1)

# Set up XGBoost parameters
params <- list(
  objective = "binary:logistic",
  eval_metric = "error",
  eta = 0.1,
  max_depth = 4,
  subsample = 0.7
)

# Train the XGBoost model
set.seed(123)
xgb_model <- xgb.train(params = params, data = train_matrix, nrounds = 100)

# Making predictions and convert them to class labels
xgb_predictions_raw <- predict(xgb_model, test_matrix)
xgb_predictions <- as.factor(ifelse(xgb_predictions_raw > 0.5, "1", "0"))

# Evaluating the model's performance
xgb_cm <- confusionMatrix(xgb_predictions, testing_set$default)
print("XGBoost Confusion Matrix:")
print(xgb_cm)