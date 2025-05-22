### Obesity Prediction: The Scale Doesn’t Lie, But Does Our Model? - Predicting obesity risk based on various lifestyle factors

#### Project Overview

This project applies supervised machine learning to classify individuals into obesity risk categories based on biometric and lifestyle data. We implemented and evaluated multiple models including logistic regression, KNN, tree-based models, and a neural network.

#### Data Sources

**Obesity Prediction Dataset**: We used the Obesity Levels Estimation Dataset. This dataset is designed for estimating obesity levels in individuals from Mexico, Peru, and Colombia based on their eating habits and physical condition. It contains 17 attributes and a total of 2,111 records. Each record is labeled with the class variable “NObesity” (Obesity Level), which categorizes individuals into seven obesity-related classifications: Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II, and Obesity Type III.
The dataset can be retrieved from [here](https://www.kaggle.com/datasets/ruchikakumbhar/obesity-prediction) and the paper [here](https://pmc.ncbi.nlm.nih.gov/articles/PMC6710633/).


#### Model Selection and Evaluation

- **Logistic Regression**: 92.2% test accuracy - served as interpretable baseline model.
- **Penalized Logistic Regression**: 93.6% test accuracy - L2 regularization showed slight improvement over standard logistic regression.
- **KNN and PCS**: Achieved 77.54% test accuracy with optimal performance at k=1, though PCA did not improve results over baseline.
- **Tree-based Models**:
  - **Baseline Decision Tree**: Exceptional 96.11% test accuracy, outperforming ensemble methods.
  - **Random Forest**: 93.62% test accuracy with 200 estimators.
  - **XGBoost**: 95.74% test accuracy with distinct feature importance patterns.
- **Neural Networks**: 83.9% test accuracy with stable training and good generalization, though less interpretable than tree-based approaches.

Note: Tree-based models' high performance was largely due to target leakage from weight and height features, with accuracy dropping significantly when these were excluded (Decision Tree: 71.6%, Random Forest: 79%, XGBoost: 79%).  

#### Key Findings
Some key findings include among others:  

- Modest Performance on Lifestyle Factors Alone: Excluding height and weight dramatically reduced accuracies (71-79%).
- Simpler Models Outperformed Complex Ones: Basic Decision Trees performed better than ensemble methods and neural networks.
- Data Quality Issues: Dataset limitations included synthetic data (77% from SMOTE) affecting accuracy metrics.
- Limited Policy Relevance: Strong predictors (height/weight) aren't modifiable and behavioral factors had weak predictive power.  

While some models performed better than others, we unveiled some data leakages from the features to the labels, which made it difficult to derive strong policy recommendations from our findings. More details in the full report.  

More information on this issue and the modelling and performance of our models can be retrieved from our final report which can be accessed here: [.pdf](https://github.com/nicolasreichardt/ml-project-obesity-prediction/blob/main/report/final_report_first.pdf), [.html](https://raw.githack.com/nicolasreichardt/ml-project-obesity-prediction/refs/heads/main/report/final_report_first.html)


#### Project Structure  

```
ML-PROJECT-OBESITY-PREDICTION/
├── notebooks/
│   ├── EDA.ipynb
│   ├── logistic_regression.ipynb
│   ├── neural_network.ipynb
│   ├── PCA_KNN.ipynb
│   ├── preprocessing.ipynb
│   ├── ridge_logistic_regression.ipynb
│   ├── train-test-split.ipynb
│   └── tree-based-models.ipynb
├── plots/
├── processed_data/
├── project_management/
├── project_report/
├── raw_data/
├── report/
│   ├── images/
│   ├── final_report_first.html
│   ├── final_report_first.md
│   └── final_report_first.pdf
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt  
```


#### Authors

- **Nadine Daum**: [E-Mail](n.daum@students.hertie-school.org), [GitHub](https://github.com/NadineDaum)
- **Ashley Razo**: [E-Mail](a.razo@students.hertie-school.org), [GitHub](https://github.com/ashley-razo)
- **Jasmin Mehnert**: [E-Mail](j.mehnert@students.hertie-school.org), [GitHub](https://github.com/jasmin-mehnert)
- **Nicolas Reichardt**: [E-Mail](n.reichardt@students.hertie-school.org), [GitHub](https://github.com/nicolasreichardt)

