import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Carica il dataset
df = pd.read_csv('datasets/ready_to_use/dataset_fifa_15_23.csv')

# 1. Analisi iniziale
print("Dimensione del dataset:", df.shape)
print("Prime righe del dataset:\n", df.head())
print("Informazioni sul dataset:\n", df.info())
print("Statistiche descrittive:\n", df.describe())

# 2. Gestione dei valori nulli
print("Valori nulli per colonna:\n", df.isnull().sum())

# Sostituisci valori nulli con la media
for column in df.select_dtypes(include=['float64', 'int64']).columns:
    df[column].fillna(df[column].mean(), inplace=True)

# 3. Normalizzazione dei dati
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 4. Analisi finale
print("Dopo il preprocessing:")
print("Valori nulli per colonna:\n", df.isnull().sum())
print("Prime righe del dataset preprocessato:\n", df.head())

# Salva il dataset preprocessato
df.to_csv('datasets/ready_to_use/dataset_fifa_15_23_preprocessed.csv', index=False)
print("Dataset preprocessato salvato come 'dataset_fifa_15_23_preprocessed.csv'")

# 5. Separazione delle caratteristiche e delle etichette
target_columns = ['age', 'overall', 'potential', 'shooting', 'passing', 'dribbling',
                  'defending', 'physic', 'attacking_crossing', 'attacking_finishing',
                  'attacking_heading_accuracy', 'attacking_short_passing',
                  'attacking_volleys', 'skill_dribbling', 'skill_curve',
                  'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
                  'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
                  'movement_reactions', 'movement_balance', 'power_shot_power',
                  'power_jumping', 'power_stamina', 'power_strength',
                  'power_long_shots', 'mentality_aggression', 'mentality_interceptions',
                  'mentality_positioning', 'mentality_vision', 'mentality_penalties',
                  'mentality_composure', 'defending_marking_awareness',
                  'defending_standing_tackle', 'defending_sliding_tackle',
                  'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
                  'goalkeeping_positioning', 'goalkeeping_reflexes', 'goalkeeping_speed']

# Assicurati che tutte le colonne target siano nel dataset
missing_targets = [col for col in target_columns if col not in df.columns]
if missing_targets:
    print(f"Attenzione: le seguenti colonne target non sono presenti nel dataset: {missing_targets}")

# Seleziona le caratteristiche e rimuovi colonne non necessarie
features = df.drop(columns=target_columns)

# 5.1 Codifica delle colonne categoriche
# Converti colonne categoriche in variabili dummy
#features = pd.get_dummies(features, drop_first=True)
target = df[target_columns]

# 6. Divisione del dataset
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 7. Modello Random Forest con Grid Search
rf = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=3, n_jobs=-1, verbose=2)

# Esecuzione della Grid Search
grid_search.fit(X_train, y_train)

# Migliori parametri trovati
print("Migliori parametri trovati:", grid_search.best_params_)

# 8. Valutazione del modello
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

# Calcolo dell'errore medio quadratico
mse = mean_squared_error(y_test, y_pred)
print(f'Error Mean Squared Error: {mse}')

# 9. Previsione per Zakaria Bakkali
zakaria_bakkali_data = df[df['long_name'] == 'Zakaria Bakkali']

if not zakaria_bakkali_data.empty:
    zakaria_bakkali_features = zakaria_bakkali_data.drop(columns=target_columns + ['long_name'])
    zakaria_bakkali_scaled = scaler.transform(zakaria_bakkali_features)
    predictions = best_rf.predict(zakaria_bakkali_scaled)

    # Mostra le previsioni
    predictions_df = pd.DataFrame(predictions, columns=target_columns)
    print("Previsioni per Zakaria Bakkali:\n", predictions_df)

    # Salva le previsioni in un file CSV
    predictions_df.to_csv('datasets/ready_to_use/predictions_zakaria_bakkali.csv', index=False)
    print("Previsioni salvate come 'predictions_zakaria_bakkali.csv'")
else:
    print("Zakaria Bakkali non trovato nel dataset.")