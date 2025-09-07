🏎 F1 Podium Predictor – Italian GP 2025






Predict the winner, podium, and top 10 finishers of the Italian Grand Prix 2025 using FastF1 session data and machine learning.

🔹 Features

Fetches FP1, FP2, FP3, and Qualifying data from FastF1.

Generates driver features (average & best lap times) for modeling.

Predicts:

🏆 Winner

🥇 Podium finishers

🔟 Top 10 finishers

Uses Random Forest Classifier for predictions.

Fully cached with FastF1 Cache for faster repeated runs.

🔹 Requirements

Python 3.9+

Packages:

pip install fastf1 pandas numpy scikit-learn

🔹 Usage

Clone the repository:

git clone https://github.com/PIYUSHhh44/F1_Podium_Predictor.git
cd F1_Podium_Predictor


Run the predictor:

python F1_podium_predictor.py


Output:

Predicted Winner

Predicted Podium

Predicted Top 10 Finishers

Example output:

🏆 Predicted Winner for Italian GP 2025:
  Driver  Position  WinnerProb
0    VER       1.0       0.68

🥇 Predicted Podium for Italian GP 2025:
  Driver  Position  PodiumProb
0    VER       1.0       0.84
1    NOR       2.0       0.82
2    PIA       3.0       0.81

🔟 Predicted Top 10 Finishers for Italian GP 2025:
  Driver  Position  Top10Prob
0    VER       1.0       0.99
1    NOR       2.0       0.97
2    PIA       3.0       0.95
3    LEC       4.0       0.97
4    HAM       5.0       0.96
5    RUS       6.0       0.99
6    ANT       7.0       0.92
7    BOR       8.0       0.89
8    ALO       9.0       0.83
9    TSU      10.0       0.82

🔹 How It Works

Fetch Session Data: Pulls live session data (practice & qualifying) using FastF1.

Feature Engineering: Calculates average & best lap times per driver.

Modeling:

Random Forest Classifier predicts podium finishers.

Separate Random Forest predicts winner.

Another model predicts top 10 finishers.

Predictions: Outputs winner, podium, and top 10 finishers sequentially by predicted finish.

🔹 Notes

FastF1 requires internet to fetch session data initially. Subsequent runs use cached data.

Predictions are based on practice and qualifying session performance.

The script handles missing lap data gracefully using heuristics.

🔹 Author

Piyush Anil Patil
B.Tech IT
Email: piyush45718@gmail.com