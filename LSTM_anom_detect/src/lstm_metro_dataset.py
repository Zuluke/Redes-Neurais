import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Carregar dados pré-processados
df = pd.read_csv('processed/preprocessed_features.csv')
window = 24

# Extrair datas, features e target
dates = pd.to_datetime(df['date_time'])
y = df['traffic_volume'].values
X = df.drop(columns=['traffic_volume','date_time']).values

# Função para criar sequências
def create_sequences(X, y, window):
    seq_X, seq_y = [], []
    for i in range(len(X) - window):
        seq_X.append(X[i:i+window])
        seq_y.append(y[i+window])
    return np.array(seq_X), np.array(seq_y)

# Criar sequências e dividir cronologicamente
total_seq, total_y = create_sequences(X, y, window)
split_idx = int(len(total_seq) * 0.7)
X_train, y_train = total_seq[:split_idx], total_y[:split_idx]
X_test,  y_test  = total_seq[split_idx:], total_y[split_idx:]

# Converter para tensores
torch.manual_seed(42)
X_train_t = torch.from_numpy(X_train).float()
y_train_t = torch.from_numpy(y_train).float().unsqueeze(1)
X_test_t  = torch.from_numpy(X_test).float()
y_test_t  = torch.from_numpy(y_test).float().unsqueeze(1)

train_ds = TensorDataset(X_train_t, y_train_t)
test_ds  = TensorDataset(X_test_t, y_test_t)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Modelo LSTM
torch.manual_seed(42)
class LSTMAnomalyDetect(nn.Module):
    def __init__(self, n_features, n_hidden=64, n_layers=2, drop=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=n_hidden,
                            num_layers=n_layers, dropout=drop, batch_first=True)
        self.fc = nn.Linear(n_hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

model = LSTMAnomalyDetect(n_features=X_train.shape[2]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Treino com early stopping
best_loss, patience, trials = np.inf, 5, 0
for epoch in range(1, 51):
    model.train()
    losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    val_pred = model(X_train_t.to(device)).detach().cpu().numpy().flatten()
    val_loss = mean_squared_error(y_train, val_pred)
    print(f"Epoch {epoch}: train MSE={np.mean(losses):.4f}, valid MSE={val_loss:.4f}")
    if val_loss < best_loss:
        best_loss = val_loss; trials = 0
        torch.save(model.state_dict(), 'best_lstm.pt')
    else:
        trials += 1
        if trials >= patience:
            print("Early stopping")
            break

# Carregar melhor modelo e avaliar
model.load_state_dict(torch.load('best_lstm.pt'))
model.eval()

# Previsões e métricas
with torch.no_grad():
    y_train_pred = model(X_train_t.to(device)).cpu().numpy().flatten()
    y_test_pred  = model(X_test_t.to(device)).cpu().numpy().flatten()

# Métricas de regressão
train_mse = mean_squared_error(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mse  = mean_squared_error(y_test, y_test_pred)
test_mae  = mean_absolute_error(y_test, y_test_pred)
print(f"TRAIN -> MSE: {train_mse:.2f}, MAE: {train_mae:.2f}")
print(f" TEST -> MSE: {test_mse:.2f}, MAE: {test_mae:.2f}")

# Threshold para detecção de anomalias a partir do erro de treino
train_errors = np.abs(y_train - y_train_pred)
thresh = train_errors.mean() + 2 * train_errors.std()
print(f"Threshold de anomalia (treino): {thresh:.2f}")

# Detectar anomalias no conjunto de teste
test_errors = np.abs(y_test - y_test_pred)
idx_anom = np.where(test_errors > thresh)[0]
print(f"Anomalias detectadas no TEST: {len(idx_anom)} de {len(y_test)} pontos")

# Plot de predição vs real e anomalias
times_test = dates.iloc[window+split_idx:].reset_index(drop=True)
plt.figure(figsize=(12,4))
plt.plot(times_test, y_test, label='True')
plt.plot(times_test, y_test_pred, label='Pred')
plt.scatter(times_test.iloc[idx_anom], y_test[idx_anom], c='red', s=10, label='Anomalia')
plt.title('Detecção de Anomalias (Teste)')
plt.xlabel('Date Time')
plt.ylabel('Volume')
plt.legend(); plt.tight_layout()
plt.savefig('plots/anomaly_detection_test.png')
plt.close()
print('Plot de anomalias de teste salvo em plots/anomaly_detection_test.png')