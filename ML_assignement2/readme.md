Student Depression Prediction: Five Classifiers Compared + Streamlit App

ML Assignment 2, M.Tech (AIML), BITS Pilani WILP.

a. Problem statement

Predict whether a university student reports depression (Yes / No) from 12 demographic, academic and lifestyle survey answers. Five classifiers (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, SVM) are trained on the same data, scored with six metrics (Accuracy, AUC, Precision, Recall, F1, MCC) and served through an interactive Streamlit app.

The labels are self-reported survey answers, not clinical diagnoses. This is a coursework project and not a screening tool.

b. Dataset description
Source: "Student Depression Detection Dataset" on Kaggle (uploaded by MKhoirul Mustaqim), originally published on Mendeley Data by Obaidul Haque and Samiul Alom.
Size: 4,464 records, 12 input features + 1 target. No missing values, 0 exact duplicate rows.
Target: Depression (Yes/No). Class balance is 73.1% No / 26.9% Yes, so the data is imbalanced.
Features (4 numeric, 8 categorical):
Feature	Type	Meaning
Gender	categorical	Gender of the respondent
Age	numeric	Age in years (17 to 28)
Family_background	categorical	Economic / class status of the family
Relationship_status	categorical	Single, In a Relationship, Complicated
Marital_status	categorical	Marital status of the student
Living_style	categorical	Living arrangement (alone/peers, family)
Current_semester	numeric	Semester currently enrolled in
CGPA	numeric	Cumulative grade point average
friend_circle	categorical	Whether the student has an active friend circle
Health_condition	categorical	General physical health (Good, Normal, Abnormal, Critical)
Cost_without_study	numeric	Estimated living costs outside educational fees
Home_division	categorical	Home region / division
Preprocessing (inside each saved pipeline, so the app needs no separate steps): median/most-frequent imputation, one-hot encoding for categorical columns, standard scaling for numeric columns (all models except the Decision Tree).
Split: stratified 80/20 (3,571 train / 893 test), seed set by RANDOM_STATE in train.py.
Model settings: Logistic Regression (max_iter=2000), Decision Tree (max_depth=6, min_samples_leaf=10), KNN (k=7), Gaussian Naive Bayes (defaults), SVM (RBF kernel, C=1, probability=True).
c. GitHub repository

TODO: paste your repository link here

bits_project/
|-- README.md               short pointer to the project folder
|-- ML_assignement2/
    |-- app.py              Streamlit app
    |-- train.py            trains and saves the five models, writes the metrics
    |-- requirements.txt
    |-- README.md           this file
    |-- test_data.csv       held-out test rows used in the experiments
    |-- metrics.csv
    |-- model/              five saved pipelines (.joblib) + meta.json

Live app: TODO: paste your Streamlit link here

d. Models used

Test-set results (893 held-out rows):

ML Model Name	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.9798	0.9954	0.9912	0.9333	0.9614	0.9485
Decision Tree	0.9843	0.9983	0.9829	0.9583	0.9705	0.9599
KNN	0.9440	0.9697	0.9358	0.8500	0.8908	0.8551
Naive Bayes	0.9429	0.9827	0.8735	0.9208	0.8966	0.8577
Support Vector Machine	0.9821	0.9989	0.9870	0.9458	0.9660	0.9542
Observations
ML Model Name	Observation about model performance
Logistic Regression	TODO: 2 to 3 sentences in your own words
Decision Tree	TODO
KNN	TODO
Naive Bayes	TODO
Support Vector Machine	TODO
Overall winner for this dataset	TODO: name it, say which metric you used to decide, and say how big the margin is
Running it
cd ML_assignement2
pip install -r requirements.txt
python train.py          # trains, evaluates and saves models + test_data.csv
streamlit run app.py     # launches the app
Limitations

Self-reported survey data from one population (the home regions suggest Bangladesh), so results should not be generalised. Several models score very high because the label is strongly tied to a few columns in this particular dataset.
