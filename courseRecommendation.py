import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple

# Machine Learning Imports
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    learning_curve
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


class StudentPerformanceAnalyzer:
    """
    A comprehensive class for analyzing student performance
    and generating course recommendations
    """

    def __init__(self, config: Dict = None):
        """
        Initialize the performance analyzer with configuration options

        :param config: Optional configuration dictionary
        """
        # Default configuration with advanced settings
        self.config = config or {
            'random_seed': 42,
            'test_size': 0.2,
            'cross_val_folds': 5,
            'model_params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'class_weight': [None, 'balanced']
            }
        }

        # System state variables
        self.raw_data = None
        self.processed_data = None
        self.features = None
        self.target = None
        self.model = None
        self.performance_metrics = {}

        # Initialize the analysis pipeline
        self._initialize_analysis()

    def _initialize_analysis(self):
        """
        Initialize the full analysis pipeline
        """
        try:
            # Attempt to load real dataset
            self.raw_data = pd.read_csv('/content/sample_data/StudentsPerformance.csv')
        except FileNotFoundError:
            # Fallback to synthetic data generation
            self.raw_data = self._generate_synthetic_data()

        # Preprocess and prepare data
        self._engineer_features()
        self._prepare_model_inputs()

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """
        Generate comprehensive synthetic student performance data

        :return: Synthetic DataFrame with student performance metrics
        """
        np.random.seed(self.config['random_seed'])
        n_samples = 1000

        return pd.DataFrame({
            'gender': np.random.choice(['male', 'female'], n_samples),
            'race/ethnicity': np.random.choice(['group A', 'group B', 'group C'], n_samples),
            'parental level of education': np.random.choice([
                'some high school', 'high school',
                'some college', 'associate\'s degree'
            ], n_samples),
            'lunch': np.random.choice(['standard', 'free/reduced'], n_samples),
            'test preparation course': np.random.choice(['completed', 'none'], n_samples),
            'math score': np.random.normal(66, 15, n_samples),
            'reading score': np.random.normal(69, 14, n_samples),
            'writing score': np.random.normal(68, 15, n_samples)
        })

    def _engineer_features(self):
        """
        Perform advanced feature engineering
        """
        # Calculate comprehensive performance metrics
        self.raw_data['total_score'] = (
                self.raw_data['math score'] +
                self.raw_data['reading score'] +
                self.raw_data['writing score']
        )
        self.raw_data['average_score'] = self.raw_data['total_score'] / 3
        self.raw_data['score_variance'] = np.var([
            self.raw_data['math score'],
            self.raw_data['reading score'],
            self.raw_data['writing score']
        ], axis=0)

    def _prepare_model_inputs(self):
        """
        Prepare input features and target variable
        """
        # Define feature categories
        self.categorical_features = [
            'gender', 'race/ethnicity',
            'parental level of education',
            'lunch', 'test preparation course'
        ]

        self.numerical_features = [
            'math score', 'reading score', 'writing score',
            'total_score', 'average_score', 'score_variance'
        ]

        # Define target variable categorization
        self.raw_data['performance_category'] = self.raw_data['total_score'].apply(
            self._categorize_performance
        )

        # Finalize features and target
        self.features = self.categorical_features + self.numerical_features
        self.target = 'performance_category'

    def _categorize_performance(self, total_score: float) -> str:
        """
        Categorize student performance based on total score

        :param total_score: Aggregated student performance score
        :return: Performance category
        """
        performance_levels = {
            'Exceptional': (240, float('inf')),
            'Advanced': (210, 240),
            'Intermediate': (180, 210),
            'Basic': (150, 180),
            'Foundational': (0, 150)
        }

        for category, (lower, upper) in performance_levels.items():
            if lower <= total_score < upper:
                return category

        return 'Foundational'

    def _create_preprocessing_pipeline(self):
        """
        Create comprehensive preprocessing pipeline

        :return: Configured preprocessing pipeline
        """
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        return ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])

    def train_predictive_model(self) -> Dict:
        """
        Train advanced predictive model with comprehensive evaluation

        :return: Performance metrics dictionary
        """
        # Prepare data
        X = self.raw_data[self.features]
        y = self.raw_data[self.target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_seed'],
            stratify=y
        )

        # Create model pipeline
        pipeline = Pipeline([
            ('preprocessor', self._create_preprocessing_pipeline()),
            ('classifier', RandomForestClassifier(
                random_state=self.config['random_seed']
            ))
        ])

        # Hyperparameter tuning
        param_grid = {
            'classifier__' + k: v
            for k, v in self.config['model_params'].items()
        }

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=self.config['cross_val_folds'],
            scoring='f1_macro'
        )

        # Fit model
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Predictions
        y_pred = best_model.predict(X_test)

        # Comprehensive metrics
        self.performance_metrics = {
            'best_params': grid_search.best_params_,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='macro'),
            'recall': recall_score(y_test, y_pred, average='macro'),
            'f1_score': f1_score(y_test, y_pred, average='macro')
        }

        # Store model
        self.model = best_model

        return self.performance_metrics

    def generate_course_recommendations(self, student_profile: Dict) -> Dict:
        """
        Generate personalized course recommendations

        :param student_profile: Dictionary of student attributes
        :return: Recommendation dictionary
        """
        # Predefined course recommendations
        course_map = {
            'Exceptional': [
                {"name": "Advanced Computer Science", "career_paths": ["AI Researcher", "Data Scientist"]},
                {"name": "Quantum Computing", "career_paths": ["Quantum Engineer", "Research Scientist"]}
            ],
            'Advanced': [
                {"name": "Software Engineering", "career_paths": ["Full Stack Developer", "DevOps Engineer"]},
                {"name": "Machine Learning", "career_paths": ["ML Engineer", "AI Specialist"]}
            ],
            'Intermediate': [
                {"name": "Web Development", "career_paths": ["Frontend Developer", "UX Designer"]},
                {"name": "Business Analytics", "career_paths": ["Business Analyst", "Product Manager"]}
            ],
            'Basic': [
                {"name": "Digital Marketing", "career_paths": ["Social Media Manager", "Content Strategist"]},
                {"name": "Project Management", "career_paths": ["Coordinator", "Junior Project Manager"]}
            ],
            'Foundational': [
                {"name": "Career Exploration", "career_paths": ["Self-Discovery", "Skills Assessment"]},
                {"name": "Foundational Skills", "career_paths": ["Personal Development", "Basic Training"]}
            ]
        }

        # Prepare input for prediction
        input_df = pd.DataFrame([student_profile])

        # Calculate total score for categorization
        input_df['total_score'] = (
                input_df['math score'] +
                input_df['reading score'] +
                input_df['writing score']
        )

        # Predict performance category
        predicted_category = self._categorize_performance(input_df['total_score'].values[0])

        return {
            'predicted_category': predicted_category,
            'recommendations': course_map.get(predicted_category, []),
            'insights': f"Based on your performance, we recommend {predicted_category} level courses."
        }

    def visualize_performance(self):
        """
        Generate comprehensive performance visualizations
        """
        plt.figure(figsize=(15, 5))

        # Performance Distribution
        plt.subplot(131)
        sns.countplot(x='performance_category', data=self.raw_data)
        plt.title('Performance Category Distribution')
        plt.xticks(rotation=45)

        # Score Correlations
        plt.subplot(132)
        score_cols = ['math score', 'reading score', 'writing score']
        sns.heatmap(self.raw_data[score_cols].corr(), annot=True, cmap='coolwarm')
        plt.title('Score Correlations')

        # Performance vs Preparation
        plt.subplot(133)
        sns.boxplot(
            x='test preparation course',
            y='total_score',
            data=self.raw_data
        )
        plt.title('Total Score by Test Prep')

        plt.tight_layout()
        plt.show()


def main():
    # Clear screen and welcome message
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

    print("===== Student Performance Course Recommendation System =====")
    print("\nWelcome! This system will help you get personalized course recommendations.")

    # Initialize analyzer
    analyzer = StudentPerformanceAnalyzer()

    # Train predictive model
    print("\n[*] Training predictive model...")
    performance_metrics = analyzer.train_predictive_model()

    # Print performance metrics
    print("\n--- Model Performance Metrics ---")
    for metric, value in performance_metrics.items():
        if isinstance(value, dict):  # Check if the value is a dictionary
            print(f"{metric.title()}: {value}")  # Print dictionary directly
        else:
            print(f"{metric.title()}: {value:.4f}")

            # Interactive user input

    def get_valid_input(prompt, options=None, input_type=str):
        """
        Helper function to validate user input
        """
        while True:
            try:
                user_input = input(prompt).strip()

                # Type conversion
                if input_type == int:
                    value = int(user_input)
                elif input_type == float:
                    value = float(user_input)
                else:
                    value = user_input

                # Optional options validation
                if options and value not in options:
                    raise ValueError(f"Invalid input. Options are: {options}")

                return value
            except ValueError as e:
                print(f"Invalid input. {e}")
                print("Please try again.")

    # Collect student profile
    student_profile = {}

    print("\n--- Student Profile Questionnaire ---")

    # Gender input
    student_profile['gender'] = get_valid_input(
        "Enter your gender (male/female): ",
        options=['male', 'female']
    ).lower()

    # Ethnicity input
    student_profile['race/ethnicity'] = get_valid_input(
        "Enter your ethnic group (group A/group B/group C): ",
        options=['group A', 'group B', 'group C']
    ).lower()

    # Parental education input
    student_profile['parental level of education'] = get_valid_input(
        "Enter parental education level (some high school/high school/some college/associate's degree): ",
        options=[
            'some high school',
            'high school',
            'some college',
            'associate\'s degree'
        ]
    ).lower()

    # Lunch type input
    student_profile['lunch'] = get_valid_input(
        "Enter lunch type (standard/free/reduced): ",
        options=['standard', 'free/reduced']
    ).lower()

    # Test preparation input
    student_profile['test preparation course'] = get_valid_input(
        "Did you complete test preparation? (completed/none): ",
        options=['completed', 'none']
    ).lower()

    # Score inputs with range validation
    student_profile['math score'] = get_valid_input(
        "Enter your math score (0-100): ",
        input_type=int,
        options=range(0, 101)
    )

    student_profile['reading score'] = get_valid_input(
        "Enter your reading score (0-100): ",
        input_type=int,
        options=range(0, 101)
    )

    student_profile['writing score'] = get_valid_input(
        "Enter your writing score (0-100): ",
        input_type=int,
        options=range(0, 101)
    )

    # Generate and display recommendations
    print("\n[*] Processing your profile and generating recommendations...")
    recommendations = analyzer.generate_course_recommendations(student_profile)

    # Display comprehensive recommendations
    print("\n===== Personalized Course Recommendations =====")
    print(f"\nPredicted Performance Category: {recommendations['predicted_category']}")
    print("\nRecommended Courses:")

    for idx, course in enumerate(recommendations['recommendations'], 1):
        print(f"\n{idx}. {course['name']}")
        print("   Potential Career Paths:")
        for path in course['career_paths']:
            print(f"   - {path}")

    print("\n--- Additional Insights ---")
    print(recommendations['insights'])

    # Visualization prompt
    view_visualization = get_valid_input(
        "\nWould you like to see performance visualizations? (yes/no): ",
        options=['yes', 'no']
    ).lower()

    if view_visualization == 'yes':
        print("\n[*] Generating Performance Visualizations...")
        analyzer.visualize_performance()

    # Farewell message
    print("\nThank you for using the Student Performance Course Recommendation System!")
    print("We hope these insights help guide your educational journey.")


if __name__ == "__main__":
    main()