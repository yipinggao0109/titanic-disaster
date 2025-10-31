import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():
    # Step 1: Load the training dataset
    print("Starting Titanic model training and evaluation...")

    data_path = os.path.join(os.path.dirname(__file__), "../data/train.csv")
    test_path = os.path.join(os.path.dirname(__file__), "../data/test.csv")

    print(f"Looking for training file at: {data_path}")
    print(f"Looking for test file at: {test_path}")

    if not os.path.exists(data_path):
        print("train.csv not found. Please make sure it is located in src/data/")
        return
    if not os.path.exists(test_path):
        print("test.csv not found. Please make sure it is located in src/data/")
        return

    train_df = pd.read_csv(data_path)
    test_df = pd.read_csv(test_path)
    print("Successfully loaded train.csv and test.csv")
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # Step 2: Explore the data
    print("\n=== Basic Info ===")
    print(train_df.info())

    print("\n=== Missing Values ===")
    print(train_df.isnull().sum().sort_values(ascending=False).head())

    # Step 3: Data Cleaning
    print("\n=== Cleaning Data ===")
    train_df["Age"].fillna(train_df["Age"].median(), inplace=True)
    train_df["Embarked"].fillna(train_df["Embarked"].mode()[0], inplace=True)
    test_df["Age"].fillna(test_df["Age"].median(), inplace=True)
    test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

    print("Filled missing values for Age, Embarked, and Fare.")

    # Step 4: Feature Engineering
    print("\n=== Encoding categorical variables ===")
    for name, df in [("train_df", train_df), ("test_df", test_df)]:
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
        df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
        dummies = pd.get_dummies(df["Embarked"], prefix="Embarked", drop_first=True)
        df.drop("Embarked", axis=1, inplace=True)
        df[dummies.columns] = dummies
        print(f"{name} encoding completed.")

    # Step 5: Drop irrelevant columns
    cols_to_drop = ["Name", "Ticket", "Cabin"]
    train_df.drop(cols_to_drop, axis=1, inplace=True)
    test_df.drop(cols_to_drop, axis=1, inplace=True)

    print(f"Dropped columns: {cols_to_drop}")
    print(f"Shape after cleaning: Train {train_df.shape}, Test {test_df.shape}")

    # Step 6: Prepare features and target
    print("\n=== Preparing data for model training ===")
    X = train_df.drop("Survived", axis=1)
    y = train_df["Survived"]

    # Align train/test columns to be identical
    X_test_final = test_df[X.columns.intersection(test_df.columns)]

    # Step 7: Train-test split for internal validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Step 8: Train logistic regression model
    print("\n=== Training Logistic Regression Model ===")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print("Model training completed.")

    # Step 9: Evaluate accuracy on training and validation set
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    # Step 10: Predict on the test set
    print("\n=== Predicting on Test Set ===")
    test_predictions = model.predict(X_test_final)
    print(f"Generated predictions for {len(test_predictions)} passengers.")

    # Step 11: Save predictions
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": test_predictions
    })
    submission_path = os.path.join(os.path.dirname(__file__), "../data/predictions.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Saved predictions to: {submission_path}")

    # Step 12: End message
    print("\nTitanic model execution completed successfully.")

if __name__ == "__main__":
    main()