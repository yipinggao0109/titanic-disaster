# titanic-disaster
Homework 3 for Northwestern MLDS400 - Titanic survival prediction using Docker

This project builds two environments — one in **Python** and one in **R** — to train and evaluate logistic regression models on the **Titanic dataset**.  
It demonstrates reproducible machine learning workflows using **Docker**, ensuring the grader can run everything from scratch with only a few simple commands.

## Repository Structure

```
titanic-disaster/
├── .gitignore
├── requirements.txt
├── Dockerfile                     ← Python container
├── src/
│   ├── data/                      ← Dataset folder (not tracked by Git)
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── predictions.csv
│   │   └── predictions_r.csv
│   ├── titanic_model/             ← Python model
│   │   └── titanic_main.py
│   └── titanic_R/                 ← R model
│       ├── Dockerfile
│       ├── install_packages.R
│       └── titanic_main.R
```

## Step 1: Download the Data

1. Go to the [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic/data).  
2. Download the following files:
   - `train.csv`
   - `test.csv`
   - `gender_submission.csv` *(optional, used for reference only)*  
3. Place them inside your local project folder:

   ```
   titanic-disaster/src/data/
   ├── train.csv
   ├── test.csv
   └── gender_submission.csv
   ```

> These files are excluded from Git tracking using `.gitignore`.

## Step 2: Run the Python Docker Container

### Build the Python Image
```bash
docker build -t titanic-app .
```

### Run the Python Container
```bash
docker run --rm -v "$PWD/src/data:/app/src/data" titanic-app
```

### Expected Output
```
Starting Titanic model training and evaluation...
Successfully loaded train.csv and test.csv
=== Cleaning Data ===
=== Training Logistic Regression Model ===
Training Accuracy: 0.8123
Validation Accuracy: 0.7891
=== Predicting on Test Set ===
Generated predictions for 418 passengers.
Saved predictions to: src/data/predictions.csv
Titanic model execution completed successfully.
```

After completion, check:
```
src/data/predictions.csv
```

## Step 3: Run the R Docker Container

### Build the R Image
```bash
docker build --platform linux/arm64 -t titanic-r-app -f src/titanic_R/Dockerfile .
```

> On Intel-based systems (Windows/Intel Mac), you can omit the `--platform linux/arm64` flag.

### Run the R Container
```bash
docker run --rm -v "$PWD/src/data:/app/src/data" titanic-r-app
```

### Expected Output
```
Starting Titanic model training and evaluation in R...
Successfully loaded train.csv and test.csv
=== Cleaning Data ===
Filled missing values for Age, Embarked, and Fare.
=== Training Logistic Regression Model ===
Model training completed.
Training Accuracy: 0.8112
Validation Accuracy: 0.7894
=== Predicting on Test Set ===
Generated predictions for 418 passengers.
Saved predictions to: src/data/predictions_r.csv
Titanic R model execution completed successfully.
```

After completion, check:
```
src/data/predictions_r.csv
```

## Step 4: Verify Results

Both models generate prediction files in `src/data/`:

| File | Description |
|------|--------------|
| `predictions.csv` | Logistic regression predictions (Python) |
| `predictions_r.csv` | Logistic regression predictions (R) |

Each file contains:
```
PassengerId,Survived
892,0
893,1
...
```

## Step 5: Clean Up (Optional)

To remove containers and free up disk space:
```bash
docker system prune -f
```

## Notes for Graders

- Both environments (Python & R) are fully containerized and reproducible.  
- Only Docker is required to run this project.  
- All print statements describe intermediate steps (data cleaning, encoding, model training).  
- The repository excludes datasets and environment files for clarity (`.gitignore` used properly).  

## Author

**Jennifer Li**  
Master of Science in Machine Learning & Data Science  
Northwestern University  
"""
