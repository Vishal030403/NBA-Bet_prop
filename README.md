# NBA Player Performance Prediction & Prop Betting Analysis

This project analyzes NBA player statistics to predict game-by-game scoring performance and identify profitable prop betting opportunities. It leverages advanced feature engineering, exploratory data analysis (EDA), and machine learning (XGBoost with Poisson regression) to model player points distributions.

## Project Overview

The core objective is to move beyond simple averages and model the probability distribution of player scoring events. By using a Poisson-based approach, the model can estimate the probability of a player scoring Over/Under a specific line, providing a quantifiable edge against market odds.

### Key Features

*   **Data Preprocessing & Cleaning:**
    *   Handles missing values and converts data types (e.g., Minutes Played from "MM:SS" to float).
    *   Creates situational features: Home/Away, Win/Loss, Days of Rest, Back-to-Back games.
*   **Advanced Feature Engineering:**
    *   **Rolling Averages & EMA:** Captures recent form (last 10 games).
    *   **Per-Minute Efficiency:** Normalizes stats by playing time.
    *   **Proxy Usage Rate:** Estimates offensive load.
    *   **Consistency Metrics:** Calculates standard deviation of points to measure volatility.
*   **Exploratory Data Analysis (EDA):**
    *   **Correlation Heatmap:** Identifies key drivers of scoring (MP, FGA, Usage).
    *   **Distribution Analysis:** Validates the Poisson distribution hypothesis for player points.
    *   **Volatility Study:** Compares consistency between different player roles (e.g., Centers vs. 3-Point Specialists).
*   **Machine Learning Model:**
    *   **Algorithm:** XGBoost Regressor.
    *   **Objective:** `count:poisson` (specifically designed for count data like points).
    *   **Evaluation:** Mean Absolute Error (MAE) and probability estimation for betting lines.

## Visualizations

The project generates several key visualizations to understand the data:

*   `eda_heatmap.png`: Correlation matrix of key metrics.
*   `eda_pts_distribution.png`: Histogram showing the right-skewed (Poisson-like) distribution of points.
*   `eda_mp_vs_pts.png`: Scatter plot confirming the strong linear relationship between Minutes Played and Points.
*   `eda_volatility_comparison.png`: KDE plot comparing the scoring consistency of Centers vs. Guards.
*   `feature_importance.png`: Bar chart showing which features most influence the model's predictions.

## Installation & Requirements

Ensure you have the following Python libraries installed:

```bash
pip install pandas numpy seaborn matplotlib xgboost scipy ydata-profiling
```

## Usage

1.  **Data:** Ensure `basic_per_game_player_stats_2013_2018.csv` is in the project directory.
2.  **Run Analysis:** Open and execute the `solution.ipynb` Jupyter Notebook.
3.  **Generate Report:** The notebook also generates a comprehensive HTML report (`report.html`) using `ydata_profiling`.

## Model Outputs

The model outputs include:
*   Projected Mean Points (Lambda).
*   Probability of scoring **OVER** a given line.
*   Probability of scoring **UNDER** a given line.

**Example Output:**
```text
Prop Pricing for Player X:
Model Projected Mean: 15.7 PTS
Market Line: 15.5
Probability of OVER: 54.3%
Probability of UNDER: 45.7%
```
