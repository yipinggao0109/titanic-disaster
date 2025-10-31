# Titanic R Model
library(readr)
library(dplyr)
library(caret)

main <- function() {
  # Step 1: Load the training dataset
  cat("Starting Titanic model training and evaluation in R...\n")

  data_path <- "src/data/train.csv"
  test_path <- "src/data/test.csv"

  cat(paste("Looking for training file at:", data_path, "\n"))
  cat(paste("Looking for test file at:", test_path, "\n"))

  if (!file.exists(data_path)) {
    stop("train.csv not found. Please make sure it is located in src/data/")
  }
  if (!file.exists(test_path)) {
    stop("test.csv not found. Please make sure it is located in src/data/")
  }

  train_df <- read_csv(data_path)
  test_df  <- read_csv(test_path)
  cat("Successfully loaded train.csv and test.csv\n")
  cat(paste("Train shape:", nrow(train_df), "rows,", ncol(train_df), "columns\n"))
  cat(paste("Test shape:", nrow(test_df), "rows,", ncol(test_df), "columns\n"))

  # Step 2: Explore the data
  cat("\n=== Basic Info ===\n")
  print(colnames(train_df))
  cat("\n=== Missing Values ===\n")
  print(colSums(is.na(train_df)))

  # Step 3: Data Cleaning
  cat("\n=== Cleaning Data ===\n")
  train_df$Age[is.na(train_df$Age)] <- median(train_df$Age, na.rm = TRUE)
  train_df$Embarked[is.na(train_df$Embarked)] <- "S"
  test_df$Age[is.na(test_df$Age)] <- median(test_df$Age, na.rm = TRUE)
  test_df$Fare[is.na(test_df$Fare)] <- median(test_df$Fare, na.rm = TRUE)
  cat("Filled missing values for Age, Embarked, and Fare.\n")

  # Step 4: Feature Engineering
  cat("\n=== Encoding categorical variables ===\n")
  train_df$Sex <- ifelse(train_df$Sex == "male", 0, 1)
  test_df$Sex  <- ifelse(test_df$Sex == "male", 0, 1)

  # One-hot encode Embarked
  for (df_name in c("train_df", "test_df")) {
    df <- get(df_name)
    df$Embarked_C <- ifelse(df$Embarked == "C", 1, 0)
    df$Embarked_Q <- ifelse(df$Embarked == "Q", 1, 0)
    df <- df %>% select(-Embarked)
    assign(df_name, df)
    cat(paste(df_name, "encoding completed.\n"))
  }

  # Step 5: Drop irrelevant columns
  cols_to_drop <- c("Name", "Ticket", "Cabin")
  train_df <- train_df %>% select(-all_of(cols_to_drop))
  test_df  <- test_df %>% select(-all_of(cols_to_drop))
  cat(paste("Dropped columns:", paste(cols_to_drop, collapse = ", "), "\n"))
  cat(paste("Shape after cleaning: Train", ncol(train_df), "columns; Test", ncol(test_df), "columns\n"))

  # Step 6: Prepare features and target
  cat("\n=== Preparing data for model training ===\n")
  y <- train_df$Survived
  X <- train_df %>% select(-Survived)
  cat(paste("Training features shape:", ncol(X), "\n"))

  # Align columns (ensure same order)
  X_test_final <- test_df[, colnames(X)]

  # Step 7: Train-test split
  set.seed(42)
  idx <- createDataPartition(y, p = 0.8, list = FALSE)
  X_train <- X[idx, ]
  y_train <- y[idx]
  X_val   <- X[-idx, ]
  y_val   <- y[-idx]
  cat(paste("Training samples:", length(y_train), ", Validation samples:", length(y_val), "\n"))

  # Step 8: Train logistic regression model
  cat("\n=== Training Logistic Regression Model ===\n")
  df_train <- cbind(Survived = y_train, X_train)
  model <- glm(Survived ~ ., data = df_train, family = binomial)
  cat("Model training completed.\n")

  # Step 9: Evaluate accuracy
  train_pred <- ifelse(predict(model, newdata = df_train, type = "response") > 0.5, 1, 0)
  df_val <- cbind(Survived = y_val, X_val)
  val_pred <- ifelse(predict(model, newdata = df_val, type = "response") > 0.5, 1, 0)
  train_acc <- mean(train_pred == y_train)
  val_acc <- mean(val_pred == y_val)
  cat(paste("Training Accuracy:", round(train_acc, 4), "\n"))
  cat(paste("Validation Accuracy:", round(val_acc, 4), "\n"))

  # Step 10: Predict on the test set
  cat("\n=== Predicting on Test Set ===\n")
  test_pred <- ifelse(predict(model, newdata = X_test_final, type = "response") > 0.5, 1, 0)
  cat(paste("Generated predictions for", length(test_pred), "passengers.\n"))

  # Step 11: Save predictions
  submission <- data.frame(
    PassengerId = test_df$PassengerId,
    Survived = test_pred
  )
  write_csv(submission, "src/data/predictions_r.csv")
  cat("Saved predictions to: src/data/predictions_r.csv\n")

  # Step 12: End message
  cat("\nTitanic R model execution completed successfully.\n")
}

main()