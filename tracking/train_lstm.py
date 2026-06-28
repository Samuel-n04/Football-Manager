import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. Définition de l'architecture du Réseau (LSTM)
class PlayerPerformanceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(PlayerPerformanceLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Couche LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Couche Linéaire de sortie pour prédire la performance future (Régression)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialisation des états cachés (h0) et de cellule (c0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Propagation en avant à travers le LSTM
        out, _ = self.lstm(x, (h0, c0))
        # On prend la sortie du dernier pas temporel de la séquence
        out = self.fc(out[:, -1, :])
        # On applique la sigmoïde et on met à l'échelle pour avoir un score entre 0 et 100
        out = torch.sigmoid(out) * 100
        return out

# 2. Pipeline de simulation et entraînement rapide
def train_performance_model():
    # Features d'entrée (4) : [Vitesse moyenne, Fatigue cumulée, Passes réussies, Tirs]
    input_dim = 4
    sequence_length = 5  # Regarder les 5 dernières séquences/matchs (Exigence sujet)
    hidden_dim = 16
    output_dim = 1      # Prédire 1 valeur : le Performance Score futur
    
    model = PlayerPerformanceLSTM(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # Génération de fausses données (Batch_size=32, Séquence=5, Features=4)
    # Remplacer ceci par vos tenseurs issus de l'historique JSON accumulé
    X_dummy = torch.randn(32, sequence_length, input_dim)
    y_dummy = torch.rand(32, output_dim) * 100  # Scores cibles entre 0 et 100
    
    # Boucle d'entraînement minimale
    model.train()
    for epoch in range(20):
        optimizer.zero_grad()
        outputs = model(X_dummy)
        loss = criterion(outputs, y_dummy)
        loss.backward()
        optimizer.step()
        
    # Sauvegarde du modèle pour l'inférence live
    torch.save(model.state_dict(), "models/player_lstm.pth")
    print("[Deep Learning] Modèle LSTM entraîné et sauvegardé avec succès.")

if __name__ == "__main__":
    train_performance_model()