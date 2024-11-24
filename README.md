## BT4012 Fraud Analytics ( Fraudulent Job Detection Project ) (Group 4)

In this project, we aim to utilize various machine learning techniques to detect fraudulent job postings and improve the safety of online job markets, especially with the rise of job scams recently.

## ⏳ Steps to setup environment before running notebook.ipynb

This project requires **Python 3.12**

```start
# start virtual env
# activate virtual env
# install dependencies

python3.12 -m venv project
source project/bin/activate
pip install -r requirements.txt
```

## Team

Feel free to contact and connect!

|                                                                                     | Name         | Github                                                                                                                                            |
| ----------------------------------------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| <img src="https://avatars.githubusercontent.com/u/80191549?v=4" width="100"></img>  | Eugene Tan   | [![Eugene Tan](https://img.shields.io/badge/GitHub-181717.svg?style=for-the-badge&logo=GitHub&logoColor=white)](https://github.com/eugenetaan)    |
| <img src="https://avatars.githubusercontent.com/u/100425549?v=4" width="100"></img> | Justin Tee   | [![Justin Tee](https://img.shields.io/badge/GitHub-181717.svg?style=for-the-badge&logo=GitHub&logoColor=white)](https://github.com/ChavChavC)     |
| <img src="https://avatars.githubusercontent.com/u/99934779?v=4" width="100"></img>  | Mervyn Seah  | [![Mervyn Seah](https://img.shields.io/badge/GitHub-181717.svg?style=for-the-badge&logo=GitHub&logoColor=white)](https://github.com/undeser)      |
| <img src="https://avatars.githubusercontent.com/u/97732408?v=4" width="100"></img>  | Zenden Leong | [![Zenden Leong](https://img.shields.io/badge/GitHub-181717.svg?style=for-the-badge&logo=GitHub&logoColor=white)](https://github.com/zendenleong) |

---

## Content Page

-   [Introduction](https://github.com/undeser/fake-jobs-detection#introduction)
-   [Dataset Description](https://github.com/undeser/fake-jobs-detection#dataset-description)
-   [Data Processing](https://github.com/undeser/fake-jobs-detection#data-processing)

---

## Introduction

Jobs are positions of employment where individuals perform specific duties, tasks, or responsibilities in exchange for compensation, such as salary, wages, or benefits. As of 2023, approximately 3.5 billion people are employed worldwide [(Statista, 2024)](https://www.statista.com/statistics/1258612/global-employment-figures/). With the advent of digital technology, a growing number of job seekers are transitioning from traditional newspaper advertisements to online job postings as their primary source of employment opportunities. Online job platforms have become indispensable tools for connecting employers and job seekers, offering unparalleled convenience and accessibility. However, this increasing reliance on digital resources has also introduced new challenges.

Fraudulent job postings are a pervasive and escalating issue in today’s digital job market, exploiting the trust of job seekers and preying on their vulnerabilities. This rise is largely fueled by the anonymity and ease of creating deceptive postings on digital platforms, coupled with the vast number of users actively seeking jobs online, which provides scammers with a broad and accessible target audience. These deceptive practices pose significant risks, ranging from financial fraud and identity theft to emotional distress, and undermine the reputation of legitimate job platforms. In 2023 alone, over 6,000 victims fell prey to job scams, resulting in losses totaling nearly S$97 million [(CNA, 2023)](https://www.channelnewsasia.com/singapore/job-scams-singapore-6000-victims-millions-dollars-losses-spf-police-3775596). With more people relying on online resources to secure employment, the prevalence of these scams erodes user confidence and creates an unsafe environment for millions of job seekers globally.

Therefore, it is crucial to address these issues, not only to prevent financial losses for job seekers but also to protect individuals from other consequences, such as identity theft. Fraudulent postings are often designed to mimic genuine opportunities, using sophisticated techniques such as advanced keyword manipulation and persuasive language to evade detection. The high volume of job postings across diverse platforms further complicates identification and prevention efforts. However, with advancements in machine learning, there is an opportunity to leverage these technologies to detect fraudulent job postings and safeguard individuals.

In this project, we aim to explore potential solutions to tackle this pressing issue by utilising machine learning techniques to detect fraudulent job postings and improve the safety of online job markets.

## Dataset Description

### Dataset Source

Our team used the Employment Scam Aegean Dataset (EMSCAD), which is a publicly available dataset containing 17,880 real-life job ads published between 2012 to 2014. The dataset aims to provide a clear picture of the Employment Scam problem to the research community and can act as a valuable testbed for scientists working on the field [(Amruthjithrajvr, n.d.)](https://www.kaggle.com/datasets/amruthjithrajvr/recruitment-scam).

We aim to make use of this dataset to train different machine learning models to identify and report fraudulent job postings before any victims fall prey to these postings.

### Dataset Description

The dataset contains 17,014 legitimate and 866 fraudulent job ads. The dataset contains 18 columns, described by the table below.

| Variable Name       | Data Type | Description                                                                              |
| :------------------ | :-------- | :--------------------------------------------------------------------------------------- |
| title               | String    | Job title.                                                                               |
| location            | String    | Geographic location of the job.                                                          |
| department          | String    | Department within the organisation.                                                      |
| salary_range        | String    | Salary range for the job (if specified).                                                 |
| company_profile     | String    | Description of the company.                                                              |
| description         | String    | Detailed description of the job role and responsibilities.                               |
| requirements        | String    | Qualifications and skills required for the job.                                          |
| benefits            | String    | Perks or benefits provided by the employer.                                              |
| telecommuting       | String    | Indicates whether telecommuting is allowed (t for yes, f for no).                        |
| has_company_logo    | String    | Indicates whether the posting includes a company logo (t for yes, f for no).             |
| has_questions       | String    | Indicates whether the posting includes screening questions (t for yes, f for no).        |
| employment_type     | String    | Type of employment (e.g., full-time, part-time, contract).                               |
| required_experience | String    | Required level of experience for the job.                                                |
| required_education  | String    | Educational qualifications required for the job.                                         |
| industry            | String    | Industry to which the job belongs.                                                       |
| function            | String    | Functional area of the job (e.g., sales, marketing).                                     |
| fraudulent          | String    | Indicates whether the job posting is fraudulent (t for yes, f for no).                   |
| in_balanced_dataset | String    | Indicates whether the observation is part of the balanced dataset (t for yes, f for no). |

### Exploratory Data Analysis (EDA) Findings

Our group performed exploratory data analysis, where the analysis aims to uncover patterns, relationships, and potential anomalies in the data, providing valuable insights to guide subsequent modelling and decision-making processes. Here are some of our key findings in table form.

| Key Finding        | Insights                                                                                       |
| :----------------- | :--------------------------------------------------------------------------------------------- |
| Imbalanced Dataset | The dataset is imbalanced, with only 4.8% of data points being fraudulent.                     |
| Education          | Up to 70% of jobs requiring "Some High School Coursework" are fraudulent job postings.         |
| Company Logo       | Job postings without a company logo have a higher chance of being fraudulent (16% vs. 2%).     |
| Entry Level        | Entry-level jobs have the highest level of fraud compared to other required experience levels. |

## Data Processing

### Pre-processing Steps

We applied several preprocessing techniques to clean and prepare the dataset for feature engineering and modelling.

Firstly, the columns with binary values ‘t’ and ‘f’ such as the fraudulent column in the initial dataset represent fraudulent and non-fraudulent postings, respectively. We converted these values into binary format (1 for fraudulent and 0 for non-fraudulent) to make them compatible with machine learning models. We also removed duplicate entries from the dataset to avoid redundancy and filled missing values in various columns to ensure consistency and completeness in the data.

Next, we performed extensive preprocessing on the text data across multiple columns. These steps included:

-   Converting text to lowercase to ensure uniformity.
-   Removing non-alphanumeric characters, punctuation and emojis to clean the data and reduce noise.
-   Removing common stopwords (e.g., "the," "and," "is") present in the text using the NLTK stopwords library to focus on meaningful words.

Lastly, we dropped columns that we deemed not to be useful as features for our model training and prediction, such as ‘in_balanced_dataset’. These preprocessing steps were essential in standardising the dataset, removing irrelevant or redundant information, and ensuring the cleaned data was ready for effective feature extraction and model training.

### Feature Engineering Steps

To enhance our models' predictive capabilities, we applied various feature engineering techniques.

Firstly, we identified that location could be a valuable feature for prediction. However, the dataset contained 3,106 unique location values, making this feature highly sparse and potentially unhelpful. To address this, we derived a new feature, country, by extracting the country from the location column. This transformation significantly reduced the number of unique values to just 91, improving the feature's usability and reducing dimensionality.

Next, we transformed categorical and boolean features into numerical formats suitable for machine learning models. For categorical features, including the newly derived country, as well as employment_type, department, and others, we applied one-hot encoding. For boolean features, such as has_company_logo, we converted True and False values into 1 and 0, respectively, ensuring compatibility with numerical models.

We also engineered new numerical features from the salary_range column. This included creating salary_lower and salary_upper to represent the range's bounds, salary_average to capture the central tendency, and salary_range_diff to quantify the range size. These features were designed to help the model identify potentially fraudulent job postings with suspicious salary ranges.

Additionally, we combined cleaned text features (title, benefits, description, requirements, company_profile) into a single feature, combined_text_data. To convert this textual data into numerical representations, we used two vectorization techniques, CountVectorizer (CountV) and Term Frequency-Inverse Document Frequency (TF-IDF). These methods allowed us to extract meaningful patterns and representations from the textual data, making it usable for model training.

### Processed Dataset

We began by transforming the feature-engineered datasets into two separate datasets. Dataset 1 includes a combination of textual, categorical, and numerical features, while Dataset 2 contains only the categorical features. Both datasets were split into training, validation, and test sets using a 70-15-15 split. The training set is used to train the models discussed in Section 4, the validation set helps detect overfitting or underfitting by assessing how well the models generalise to unseen data, and the test set is reserved for evaluating final model performance.

By training models on the comprehensive feature set (Dataset 1) and the simpler categorical-only feature set (Dataset 2), we aim to determine whether a simpler model, without incorporating textual or numerical features, can achieve comparable performance. This approach serves as a form of feature importance analysis, enabling us to evaluate whether a streamlined model may generalise well while reducing the complexity associated with text and numerical data preprocessing.

For Dataset 1, which includes the combined text column, we employ two distinct methods for text vectorization: 1a) CountVectorizer (CountV) and 1b) TF-IDF. Hence, we will train the models separately on three different datasets: Dataset 1a, Dataset 1b, and Dataset 2. This multi-dataset approach allows us to assess the effectiveness of different text vectorization methods on model performance and compare them against a simpler categorical-only model.

| Dataset Names | Columns                                                             | Feature Columns                          | Target Column                      |
| :------------ | :------------------------------------------------------------------ | :--------------------------------------- | :--------------------------------- |
| Dataset 1a    | Categorical, Numerical, Count Vectorized Embedding for textual data | X_train_tfidf, X_val_tfidf, X_test_tfidf | y_train, y_val, y_test             |
| Dataset 1b    | Categorical, Numerical, TF-IDF Embedding for textual data           | X_train_count, X_val_count, X_test_count | y_train, y_val, y_test             |
| Dataset 2     | Categorical columns only                                            | X_train_cat, X_val_cat, X_test_cat       | y_train_cat, y_val_cat, y_test_cat |
