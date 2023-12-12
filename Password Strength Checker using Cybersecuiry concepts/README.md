# Machine Learning-based Password Strength Checker

## Project Overview
This project is aimed at developing a machine learning model capable of assessing the strength of passwords. The motivation behind this project is to enhance cybersecurity by providing a robust means of evaluating password complexity and resistance against potential breaches.

## Code Description
The project utilizes Python with libraries such as `pandas`, `scikit-learn`, and `seaborn` for data manipulation, model training, evaluation, and visualization.

The primary steps involved in the project include:
- Reading and preprocessing data from a CSV file.
- Feature extraction from passwords using TF-IDF vectorization.
- Applying a machine learning model (Logistic Regression) to classify passwords into strength categories.
- Using k-fold cross-validation to evaluate the model's performance.
- Generating and visualizing a confusion matrix to understand the model's predictions.

## Data Preprocessing
Passwords are read from a CSV file, where each password may contain commas. The preprocessing step involves combining all fields before the last comma into the password and treating the last field as the password's strength category.

## Model Training and Evaluation
The Logistic Regression model is trained using a k-fold cross-validation approach to ensure the model's robustness and to prevent overfitting. The classification report and confusion matrix for the last fold are used to assess the model's performance.

## Results
The model's performance is evaluated using accuracy, precision, recall, and F1-score. The confusion matrix provides a detailed view of the true positive, false positive, false negative, and true negative predictions across the strength categories (0 for weak, 1 for medium, 2 for strong).

- **Average Accuracy**: The model achieves an average accuracy of 86%, indicating its effectiveness in classifying password strengths.
- **Classification Report**: The report for the last fold includes metrics such as precision, recall, and F1-score for each class, providing insight into the model's predictive capabilities.
- **Confusion Matrix**: The matrix visualizes the model's predictions, with a majority of correct predictions along the diagonal.

## Visualization
The confusion matrix for the last fold of cross-validation is visualized using a heatmap, highlighting the distribution of predictions across actual and predicted classes.
![Figure](https://github.com/JPL11/Machine-Learning/assets/86073892/5e7b6224-4af2-48c8-bd1a-dc190d6bd052)
![ClassificationReport](https://github.com/JPL11/Machine-Learning/assets/86073892/5dbab857-5124-49e7-b0c0-6efae31fb97a)


## Repository Contents
- `PasswordStrengthChecker.py`: The main Python script with the project's code.
- `Machine Learning-based Password Strength Checker.pdf`: Presentation slides summarizing the project.
- `Project description.pdf`: The original project description and requirements.
- `Figure.png`: The generated confusion matrix visualized as a heatmap.
- `ClassificationReport.png`: The generated classification using Logistic Regression visualized.

## Usage
To run this project:
1. Ensure you have Python and the required libraries installed.
2. Clone this repository.
3. Run the script `PasswordStrengthChecker.py`.

## Contributions
Contributions to this project are welcome. Please open an issue to discuss proposed changes or submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

