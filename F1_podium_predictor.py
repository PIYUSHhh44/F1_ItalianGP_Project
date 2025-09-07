import fastf1
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Enable cache
fastf1.Cache.enable_cache('f1_cache')


def load_session_data(year, event, session_type):
    """Load FastF1 session and return lap data + results."""
    session = fastf1.get_session(year, event, session_type)
    session.load()
    return session


def get_practice_features(year, event):
    features = []
    for fp in ['FP1', 'FP2', 'FP3']:
        try:
            session = load_session_data(year, event, fp)
            laps = session.laps

            if 'LapTime' not in laps.columns or laps.empty:
                print(f"⚠️ No lap data available for {fp}")
                continue

            # average + best lap per driver
            avg_laps = (
                laps.groupby('Driver')['LapTime']
                .mean()
                .dt.total_seconds()
                .reset_index()
                .rename(columns={'LapTime': f'{fp}_avg'})
            )
            best_laps = (
                laps.groupby('Driver')['LapTime']
                .min()
                .dt.total_seconds()
                .reset_index()
                .rename(columns={'LapTime': f'{fp}_best'})
            )

            df = avg_laps.merge(best_laps, on="Driver", how="inner")
            features.append(df)
        except Exception as e:
            print(f"Could not load {fp}: {e}")

    if not features:
        return pd.DataFrame()

    final = features[0]
    for df in features[1:]:
        final = final.merge(df, on="Driver", how="outer")

    return final


# 1. Fetch 2025 Italian GP Data

year = 2025
event = 'Italian Grand Prix'

practice_df = get_practice_features(year, event)

# Qualifying
quali_session = load_session_data(year, event, 'Q')
quali_results = quali_session.results
quali_df = quali_results[['Abbreviation', 'Position']].copy()
quali_df.rename(columns={'Abbreviation': 'Driver'}, inplace=True)

# Merge practice + quali
data = quali_df.merge(practice_df, on="Driver", how="left")
data = data.fillna(data.mean(numeric_only=True))


# 2. Podium Model

data['Podium'] = data['Position'].apply(lambda x: 1 if x <= 3 else 0)

X = data.drop(columns=['Driver', 'Podium'])
y = data['Podium']

if len(set(y)) < 2:
    print("⚠️ Only one class found in podium labels — falling back to heuristic.")
    data['PodiumProb'] = 1 / data['Position']
else:
    podium_model = RandomForestClassifier(n_estimators=200, random_state=42)
    podium_model.fit(X, y)
    data['PodiumProb'] = podium_model.predict_proba(X)[:, 1]


# 3. Winner Model

data['Winner'] = data['Position'].apply(lambda x: 1 if x == 1 else 0)
y_winner = data['Winner']

if len(set(y_winner)) < 2:
    print("⚠️ Only one class found in winner labels — falling back to heuristic.")
    data['WinnerProb'] = 1 / data['Position']
else:
    winner_model = RandomForestClassifier(n_estimators=300, random_state=42)
    winner_model.fit(X, y_winner)
    data['WinnerProb'] = winner_model.predict_proba(X)[:, 1]


# 4. Top 10 Model

data['Top10'] = data['Position'].apply(lambda x: 1 if x <= 10 else 0)
y_top10 = data['Top10']

if len(set(y_top10)) < 2:
    print("⚠️ Only one class found in top10 labels — falling back to heuristic.")
    data['Top10Prob'] = 1 / data['Position']
else:
    top10_model = RandomForestClassifier(n_estimators=250, random_state=42)
    top10_model.fit(X, y_top10)
    data['Top10Prob'] = top10_model.predict_proba(X)[:, 1]


# 5. Final Predictions

# Winner
winner_pred = data.sort_values('WinnerProb', ascending=False).head(1)

# Podium (sorted by position order)
podium_preds = data[data['Position'] <= 3].sort_values('Position')

# Top 10 (sorted by position order)
top10_preds = data[data['Position'] <= 10].sort_values('Position')


# 6. Print Results

print("\n🏆 Predicted Winner for Italian GP 2025:")
print(winner_pred[['Driver', 'Position', 'WinnerProb']])

print("\n🥇 Predicted Podium for Italian GP 2025:")
print(podium_preds[['Driver', 'Position', 'PodiumProb']])

print("\n🔟 Predicted Top 10 Finishers for Italian GP 2025:")
print(top10_preds[['Driver', 'Position', 'Top10Prob']])

print("\n🏁 Final Predicted Standings for Italian GP 2025:")
for _, row in top10_preds.iterrows():
    print(f"{int(row['Position'])}. {row['Driver']}")
