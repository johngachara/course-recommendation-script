# **Student Performance Course Recommendation System**

## **Overview**
The **Student Performance Course Recommendation System** is a machine learning powered tool designed to analyze student performance data and recommend suitable courses and career paths. By leveraging machine learning, the system categorizes students into performance levels based on their academic scores and provides actionable recommendations and insights.

---

## **Features**
- **Personalized Course Recommendations:** Tailored to individual student profiles.
- **Performance Analysis:** Includes visualizations for score distributions, correlations, and key insights.
- **Interactive User Input:** Allows users to input their profile and receive recommendations instantly.
- **Optimized Model Training:** Hyperparameter tuning using GridSearchCV for improved performance.

---

## **Model and Dataset**
- **Model:** Random Forest Classifier.
- **Dataset:** The system is trained on the **Students Performance Dataset** from Kaggle. A sample CSV file is included in the project files.
- **Training Observations:** 
  - The model performs better than expected, with robust metrics across different evaluations.
  - Metrics achieved:
    - **Accuracy:** 0.9900
    - **Precision:** 0.9882
    - **Recall:** 0.8667
    - **F1 Score:** 0.8939

---

## **Requirements**
### **Dependencies**
- **Python Version:** 3.8+
- Install dependencies using:
  ```bash
  pip install -r requirements.txt
  
# Key Libraries Used:


numpy

pandas

matplotlib

seaborn

scikit-learn

# System Recommendations
Run Environment: 

Due to the computational demands of GridSearchCV during hyperparameter tuning:
Google Colab is recommended for execution.

Upload the StudentsPerformance.csv file provided with the project to the Colab environment when prompted.

How to Run

# On Google Colab
Open the attached ipynb file which should take you to a colab session,upload the student perfomance csv file ,it will still work even if you dont upload it since the script has a fallback of creating a sample dataset incase one is not provided.

# Locally

Clone the repository:

git clone https://github.com/johngachara/course-recommendation-script

cd course-recommendation-script

Install dependencies:

pip install -r requirements.txt

Run the script:

python courseRecommendation.py

# Interactive Session

Follow the on-screen prompts to input student profile details.

View personalized course recommendations and performance insights.

Optionally, explore performance visualizations.

# Visualizations

Performance Category Distribution: Visualize the distribution of students across categories.

Score Correlations: Explore relationships between math, reading, and writing scores.

Test Preparation vs. Total Score: Analyze the impact of test preparation on academic performance.

# Project Details

Dataset: Students Performance Dataset on Kaggle.

Developed for: Expert Systems Unit Assignment.

The model demonstrated high accuracy and precision but slightly lower recall, indicating room for improvement in identifying all instances of specific categories.

For optimal performance, consider running the script in Google Colab with adequate computational resources.

