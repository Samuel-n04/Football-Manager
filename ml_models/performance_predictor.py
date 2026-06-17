import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
import pickle
import os

# Optional deep-learning backend (torch-based LSTM via a simple NumPy impl as fallback)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

SEQ_LEN = 5  # number of past matches used as input
FEATURES = ['SpeedAvg', 'FatigueScore', 'PassAccuracy', 'Shots', 'Distance']


class PerformancePredictor:
    """
    Prédit la performance future d'un joueur.

    Modèles :
        - MLP  (sklearn)
        - LSTM (PyTorch si disponible, sinon simulation NumPy)
    """

    def __init__(self):
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                                random_state=42, early_stopping=True)
        self.lstm: '_SimpleLSTM | None' = None
        self.is_trained = False
        self.results: dict = {}

    # ── public API ─────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> dict:
        """
        df must contain PlayerID, MatchID, and feature columns.
        Builds sequences of SEQ_LEN consecutive matches per player.
        """
        X_seq, y_seq, X_flat, y_flat = self._build_sequences(df)

        if len(X_flat) < 10:
            self.results = {'error': 'Not enough data (need at least 10 sequences).'}
            return self.results

        X_flat_scaled = self.scaler_X.fit_transform(X_flat)
        y_flat_scaled = self.scaler_y.fit_transform(y_flat.reshape(-1, 1)).ravel()

        split = int(0.8 * len(X_flat_scaled))
        X_tr, X_te = X_flat_scaled[:split], X_flat_scaled[split:]
        y_tr, y_te = y_flat_scaled[:split], y_flat_scaled[split:]

        # MLP
        self.mlp.fit(X_tr, y_tr)
        mlp_pred_scaled = self.mlp.predict(X_te)
        mlp_pred = self.scaler_y.inverse_transform(mlp_pred_scaled.reshape(-1, 1)).ravel()
        mlp_true = self.scaler_y.inverse_transform(y_te.reshape(-1, 1)).ravel()
        mlp_mae = mean_absolute_error(mlp_true, mlp_pred)

        # LSTM
        lstm_mae = None
        if TORCH_AVAILABLE and len(X_seq) >= 10:
            # Use a dedicated per-timestep scaler (5 features) separate from the MLP scaler (25)
            scaler_seq = MinMaxScaler()
            X_seq_scaled = scaler_seq.fit_transform(X_seq.reshape(-1, X_seq.shape[-1]))
            X_seq_scaled = X_seq_scaled.reshape(X_seq.shape)
            y_seq_scaled = self.scaler_y.transform(y_seq.reshape(-1, 1)).ravel()

            s2 = int(0.8 * len(X_seq_scaled))
            self.lstm = _SimpleLSTM(input_size=X_seq.shape[-1])
            lstm_mae = self.lstm.fit(
                X_seq_scaled[:s2], y_seq_scaled[:s2],
                X_seq_scaled[s2:], y_seq_scaled[s2:],
                self.scaler_y
            )

        self.is_trained = True
        self.results = {
            'mlp_mae': round(mlp_mae, 4),
            'lstm_mae': round(lstm_mae, 4) if lstm_mae is not None else 'N/A (PyTorch absent)',
            'sequences_built': len(X_flat),
            'best_model': 'LSTM' if (lstm_mae and lstm_mae < mlp_mae) else 'MLP',
        }
        return self.results

    def predict_next(self, player_history: pd.DataFrame) -> float:
        """Predict next-match performance from the last SEQ_LEN rows of player_history."""
        if not self.is_trained:
            raise RuntimeError("Train first with .train(df).")
        avail = [f for f in FEATURES if f in player_history.columns]
        X = player_history[avail].tail(SEQ_LEN).values
        if len(X) < SEQ_LEN:
            X = np.pad(X, ((SEQ_LEN - len(X), 0), (0, 0)), mode='edge')
        X_flat = X.reshape(1, -1)
        X_scaled = self.scaler_X.transform(X_flat)
        pred_scaled = self.mlp.predict(X_scaled)
        pred = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        return round(float(np.clip(pred, 0, 100)), 2)

    def save(self, path: str = 'stubs/performance_predictor.pkl') -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str = 'stubs/performance_predictor.pkl') -> 'PerformancePredictor':
        with open(path, 'rb') as f:
            return pickle.load(f)

    # ── private helpers ─────────────────────────────────────────────────────

    def _build_sequences(self, df: pd.DataFrame):
        avail = [f for f in FEATURES if f in df.columns]
        target_col = 'PerformanceScore'
        sequences, targets = [], []

        for pid in df['PlayerID'].unique():
            player_df = df[df['PlayerID'] == pid].sort_values('MatchID')
            vals = player_df[avail].values
            perf = player_df[target_col].values
            for i in range(len(vals) - SEQ_LEN):
                sequences.append(vals[i:i + SEQ_LEN])
                targets.append(perf[i + SEQ_LEN])

        if not sequences:
            return (np.array([]), np.array([]),
                    np.array([]), np.array([]))

        X_seq = np.array(sequences, dtype=np.float32)
        y_seq = np.array(targets, dtype=np.float32)
        X_flat = X_seq.reshape(len(X_seq), -1)
        return X_seq, y_seq, X_flat, y_seq


# ── minimal PyTorch LSTM wrapper ─────────────────────────────────────────────

class _SimpleLSTM:
    def __init__(self, input_size: int, hidden: int = 32, epochs: int = 50, lr: float = 1e-3):
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.input_size = input_size
        self._build(input_size)

    def _build(self, input_size):
        import torch.nn as nn
        class _Net(nn.Module):
            def __init__(self, inp, hid):
                super().__init__()
                self.lstm = nn.LSTM(inp, hid, batch_first=True)
                self.fc = nn.Linear(hid, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        self.net = _Net(input_size, self.hidden)

    def fit(self, X_tr, y_tr, X_te, y_te, scaler_y) -> float:
        import torch
        import torch.nn as nn

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        Xtr_t = torch.tensor(X_tr, dtype=torch.float32)
        ytr_t = torch.tensor(y_tr, dtype=torch.float32)
        Xte_t = torch.tensor(X_te, dtype=torch.float32)
        yte_t = torch.tensor(y_te, dtype=torch.float32)

        self.net.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = criterion(self.net(Xtr_t), ytr_t)
            loss.backward()
            optimizer.step()

        self.net.eval()
        with torch.no_grad():
            preds_scaled = self.net(Xte_t).numpy()

        preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
        true = scaler_y.inverse_transform(yte_t.numpy().reshape(-1, 1)).ravel()
        return float(mean_absolute_error(true, preds))

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.tensor(X, dtype=torch.float32)).numpy()
