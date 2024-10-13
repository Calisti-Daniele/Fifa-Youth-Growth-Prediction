import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

# Percorso del file dove verrà salvato il modello
model_path = '../models/random_forest_multioutput_model.pkl'

# Creazione del DataFrame
df = pd.read_csv('../datasets/ready_to_use/dataset_fifa_15_23_preprocessed.csv')

# Separazione delle colonne informative che non devono essere incluse nel modello
cols_to_keep = ['player_url', 'short_name', 'long_name', 'player_positions', 'club_name', 'league_name',
                'nationality_name', 'preferred_foot']
informative_data = df[cols_to_keep]

# Colonne target
cols_targets = ['overall', 'potential', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
                'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy',
                'attacking_short_passing', 'attacking_volleys', 'skill_dribbling', 'skill_curve',
                'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
                'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
                'movement_reactions', 'movement_balance', 'power_shot_power', 'power_jumping',
                'power_stamina', 'power_strength', 'power_long_shots', 'mentality_aggression',
                'mentality_interceptions', 'mentality_positioning', 'mentality_vision',
                'mentality_penalties', 'mentality_composure', 'defending_marking_awareness',
                'defending_standing_tackle', 'defending_sliding_tackle', 'goalkeeping_diving',
                'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning',
                'goalkeeping_reflexes', 'goalkeeping_speed']

# Rimuovere le colonne informative dal set di feature per il modello
X = df.drop(columns=cols_to_keep)
y = df[cols_targets]  # Le variabili target

# Divisione del dataset in train e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verifica se il modello esiste già
if os.path.exists(model_path):
    # Caricamento del modello esistente
    model = joblib.load(model_path)
    print("Modello caricato.")
else:
    # Creazione e addestramento del modello
    model = MultiOutputRegressor(RandomForestRegressor())
    model.fit(X_train, y_train)

    # Salvataggio del modello addestrato
    os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Crea la cartella se non esiste
    joblib.dump(model, model_path)
    print("Modello addestrato e salvato.")

# Calcolo delle predizioni
y_pred = model.predict(X_test)

# Calcolo delle importanze delle feature (per il primo output)
importances = model.estimators_[0].feature_importances_
feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

# Visualizzazione delle importanze delle feature
print(feature_importances)
