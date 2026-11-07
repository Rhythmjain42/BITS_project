# Student Depression Prediction: Five Classifiers Compared + Streamlit App

ML Assignment 2, M.Tech (AIML), BITS Pilani WILP.

## a. Problem statement

Predict whether a university student reports depression (**Yes / No**) from 12 demographic, academic and lifestyle survey answers. Five classifiers (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, SVM) are trained on the same data and scored with six metrics (Accuracy, AUC, Precision, Recall, F1, MCC). The saved models are served through an interactive Streamlit app.

The labels are **self-reported survey answers, not clinical diagnoses**. This is a coursework project and not a screening tool.

## b. Dataset description

- **Source:** "Student Depression Detection Dataset" on Kaggle (uploaded by MKhoirul Mustaqim), originally published on Mendeley Data by Obaidul Haque and Samiul Alom.
- **Size:** 4,464 records, 12 input features + 1 target. No missing values and 0 exact duplicate rows.
- **Target:** `Depression` (Yes/No). The class balance is 73.1% No and 26.9% Yes, so the data is imbalanced.
- **Features (4 numeric, 8 categorical):**

| Feature | Type | Meaning |
|---|---|---|
| Gender | categorical | Gender of the respondent |
| Age | numeric | Age in years (17 to 28) |
| Family_background | categorical | Economic / class status of the family |
| Relationship_status | categorical | Single, In a Relationship, Complicated |
| Marital_status | categorical | Marital status of the student |
| Living_style | categorical | Living arrangement (alone/peers, family) |
| Current_semester | numeric | Semester currently enrolled in |
| CGPA | numeric | Cumulative grade point average |
| friend_circle | categorical | Whether the student has an active friend circle |
| Health_condition | categorical | General physical health (Good, Normal, Abnormal, Critical) |
| Cost_without_study | numeric | Estimated living costs outside educational fees |
| Home_division | categorical | Home region / division |

- **Preprocessing** (inside each saved pipeline, so the app needs no separate steps): median / most-frequent imputation, one-hot encoding for categorical columns, standard scaling for numeric columns (all models except the Decision Tree).
- **Split:** stratified 80/20 (3,571 train rows, 893 test rows). The seed is `RANDOM_STATE` in `train.py`.
- **Prediction rule:** a student is predicted "Depression" when the model's probability is 0.50 or higher. The app lets you move this threshold.
- **Model settings:** Logistic Regression (`max_iter=2000`), Decision Tree (`max_depth=6`, `min_samples_leaf=10`), KNN (`k=7`), Gaussian Naive Bayes (defaults), SVM (RBF kernel, `C=1`, `probability=True`).

## c. GitHub repository

**GitHub link:** (https://github.com/Rhythmjain42/BITS_project/tree/main/ML_assignement2)

**Live Streamlit app:** (https://bitsproject-ihg6hquruqscm9fayygsto.streamlit.app/)

```
bits_project/
|-- README.md               
|-- ML_assignement2/
    |-- app.py              Streamlit app
    |-- train.py            trains and saves the five models, writes the metrics
    |-- requirements.txt
    |-- README.md           this file
    |-- test_data.csv       held-out test rows used in the experiments
    |-- metrics.csv
    |-- model/              five saved pipelines (.joblib) + meta.json
```

## d. Models used

Test-set results on the 893 held-out rows:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9798 | 0.9954 | 0.9912 | 0.9333 | 0.9614 | 0.9485 |
| Decision Tree | 0.9843 | 0.9983 | 0.9829 | 0.9583 | 0.9705 | 0.9599 |
| KNN | 0.9440 | 0.9697 | 0.9358 | 0.8500 | 0.8908 | 0.8551 |
| Naive Bayes | 0.9429 | 0.9827 | 0.8735 | 0.9208 | 0.8966 | 0.8577 |
| Support Vector Machine | 0.9866 | 0.9989 | 0.9872 | 0.9625 | 0.9747 | 0.9657 |

There are 240 depressed and 653 non-depressed students in the test set.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very high precision (0.9912): it raised only 2 false alarms out of 653 non-depressed students. The price is recall of 0.9333, so it missed 16 of the 240 depressed students. It only says "depressed" when it is quite sure. A simple linear model doing this well suggests the main signals in the data add up in a fairly straight way. |
| Decision Tree | Strong all round (accuracy 0.9843, MCC 0.9599). Recall of 0.9583 is better than Logistic Regression, KNN and Naive Bayes. Its top features were Health_condition Critical (0.487), Health_condition Abnormal (0.216) and Relationship_status Single (0.158), together about 86% of the importance. So the label depends on a few columns, and a small tree with depth 6 can catch that. It is also the easiest model to explain. |
| KNN | Weakest on recall (0.8500): about 36 of 240 depressed students were missed. Accuracy is 0.9440 and MCC is 0.8551. A likely reason is that most columns are one-hot encoded, so the distance mixes many columns that do not matter with the few that do, and the 7 neighbours are not always the right ones. This is my reading, I did not test it. |
| Naive Bayes | Lowest precision (0.8735): 32 false alarms, plus 19 missed students. MCC is 0.8577. Its AUC is still high (0.9827), which means it ranks students well but its 0.50 cut-off is off. It assumes the features are independent and Gaussian, and one-hot columns such as the Health_condition ones do not fit that. |
| Support Vector Machine | Best on all six metrics: accuracy 0.9866, AUC 0.9989, MCC 0.9657. It made only 12 mistakes on 893 rows (3 false alarms, 9 missed students). The RBF kernel can combine the important columns in a non-linear way, which the linear model cannot. |
| **Overall winner for this dataset** | **Support Vector Machine.** I decided by MCC because the classes are imbalanced (27% depressed), and the SVM is also first on the other five metrics. The margin is small: the Decision Tree and Logistic Regression are close, and the SVM beats the tree by only a couple of students on 893 rows, on one train/test split. If explaining the model matters more than the last bit of accuracy, the Decision Tree is a sensible choice. |

All five models score high because the label depends strongly on a few columns in this dataset. These results should not be taken as the ability to detect depression in real students.
