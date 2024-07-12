# Video Game Sales Anomaly Detection

This project focuses on detecting anomalies in video game sales data using machine learning techniques. Anomalies are identified based on sales patterns across different regions using Isolation Forest and One-Class SVM algorithms.

## Dataset

The dataset used for this project contains information about video game sales across various regions, including North America, Europe, Japan, and others. Each entry includes details such as platform, genre, publisher, and sales figures.

### Columns:
- **Rank**: Ranking of the game.
- **Name**: Name of the game.
- **Platform**: Gaming platform (e.g., PC, PS4, Xbox).
- **Year**: Year of release.
- **Genre**: Game genre (e.g., Action, Sports).
- **Publisher**: Publisher of the game.
- **NA_Sales**: Sales in North America (in millions).
- **EU_Sales**: Sales in Europe (in millions).
- **JP_Sales**: Sales in Japan (in millions).
- **Other_Sales**: Sales in other regions (in millions).
- **Global_Sales**: Total global sales (in millions).

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/merttcetn/video-games-sales-anomaly-detection.git
   cd video-game-sales-anomaly-detection
   ```

2. **Install dependencies**:
   - Ensure you have Python 3.x and pip installed.
   - Install required libraries:
     ```bash
     pip install pandas scikit-learn seaborn matplotlib
     ```

3. **Run the code**:
   - Execute the main script to preprocess the data, train anomaly detection models, and visualize results:
     ```bash
     python video_game_sales_anomaly_detection.py
     ```

## Approach

1. **Data Preprocessing**:
   - Missing values are removed.
   - Categorical features (`Platform`, `Genre`, `Publisher`) are encoded using `LabelEncoder`.
   - Numerical features (`NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales`, `Global_Sales`) are normalized using `StandardScaler`.

2. **Anomaly Detection**:
   - **Isolation Forest**: Unsupervised learning algorithm that identifies anomalies based on isolation of instances in trees.
   - **One-Class SVM**: An SVM algorithm that learns a decision function for novelty detection (anomaly detection).

3. **Visualization**:
   - Pair plots and correlation heatmaps are used to visualize the distribution and relationships in the sales data.
   - Scatter plots highlight anomalies detected by Isolation Forest and One-Class SVM.

4. **Results**:
   - An "Anomaly" column is added to the dataset to indicate detected anomalies (`True` for anomaly, `False` otherwise).
   - Summary statistics and visualizations help interpret and understand the anomalies identified.

## Results

- Anomalies detected by Isolation Forest and One-Class SVM are visualized using scatter plots.
- Summary statistics include the number of anomalies detected and a sample of the detected anomalies.

## Future Improvements

- Explore different anomaly detection algorithms to compare performance.
- Enhance visualization techniques for more insightful anomaly interpretation.
- Handle imbalanced data scenarios to improve anomaly detection accuracy.
