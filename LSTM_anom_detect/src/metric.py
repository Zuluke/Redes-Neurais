import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Carregar previsões e verdadeiros
df = pd.read_csv('results/full_test_set.csv')
# Supondo colunas 'true' e 'pred' no conjunto de teste
true = df['true'].values
pred = df['pred'].values

# Calcular métricas
mse = mean_squared_error(true, pred)
mae = mean_absolute_error(true, pred)
rmse = np.sqrt(mse)

# Exibir resultados
print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Salvar em CSV
metrics_df = pd.DataFrame({
    'metric': ['MSE','MAE','RMSE'],
    'value': [mse, mae, rmse]
})
metrics_df.to_csv('results/metrics.csv', index=False)
print('Métricas salvas em results/metrics.csv')
