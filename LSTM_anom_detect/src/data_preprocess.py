import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Criar pastas de saída
os.makedirs('processed', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# 1. Fetch dataset Metro Interstate Traffic Volume (id=492)
print('Carregando dataset...')
dataset = fetch_ucirepo(id=492)
df = dataset.data.features.copy()
df['traffic_volume'] = dataset.data.targets.astype(int)

df['date_time'] = pd.to_datetime(df['date_time'])
df.sort_values('date_time', inplace=True)
df.reset_index(drop=True, inplace=True)

# 2. Salvar CSV completo
df.to_csv('./processed/traffic_full.csv', index=False)
print('CSV completo salvo em processed/traffic_full.csv')

# 3. EDA: plot série completa
plt.figure(figsize=(12,4))
plt.plot(df['date_time'], df['traffic_volume'])
plt.title('Traffic Volume Over Time')
plt.xlabel('Date Time')
plt.ylabel('Volume')
plt.tight_layout()
plt.savefig('./plots/traffic_full.png')
plt.close()
print('Plot completo salvo em plots/traffic_full.png')

# 4. Plot por ano
for year, grp in df.groupby(df['date_time'].dt.year):
    plt.figure(figsize=(12,4))
    plt.plot(grp['date_time'], grp['traffic_volume'])
    plt.title(f'Traffic Volume in {year}')
    plt.xlabel('Date Time')
    plt.ylabel('Volume')
    plt.tight_layout()
    path = f'./plots/traffic_{year}.png'
    plt.savefig(path)
    plt.close()
    print(f'Plot {year} salvo em {path}')

# 4b. Plot por mês (ano-mês)
for (year, month), grp in df.groupby([df['date_time'].dt.year, df['date_time'].dt.month]):
    plt.figure(figsize=(12,4))
    plt.plot(grp['date_time'], grp['traffic_volume'])
    plt.title(f'Traffic Volume in {year}-{month:02d}')
    plt.xlabel('Date Time')
    plt.ylabel('Volume')
    plt.tight_layout()
    path = f'./plots/traffic_{year}_{month:02d}.png'
    plt.savefig(path)
    plt.close()
    print(f'Plot {year}-{month:02d} salvo em {path}')

# 5. Anomalias via Z-score univariada (threshold=2)
traffic = df['traffic_volume'].values
z_scores = np.abs(stats.zscore(traffic))
thresh = 2
anomaly_df = df[z_scores > thresh].copy()
anomaly_df.to_csv('./processed/anomalies_zscore.csv', index=False)
print('Anomalias Z-score salvas em processed/anomalies_zscore.csv')

# 6. Pré-processamento multivariado
features = ['holiday','temp','rain_1h','snow_1h','clouds_all','weather_main']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', sparse_output=False), ['holiday','weather_main']),
    ('num', MinMaxScaler(), ['temp','rain_1h','snow_1h','clouds_all'])
])
X_all = preprocessor.fit_transform(df[features])
y_all = df['traffic_volume'].values

# Montar DataFrame para CSV preprocessed
cols = preprocessor.get_feature_names_out()
df_proc = pd.DataFrame(X_all, columns=cols)
df_proc['traffic_volume'] = y_all
df_proc['date_time'] = df['date_time'].values
df_proc.to_csv('./processed/preprocessed_features.csv', index=False)
print('Recursos pré-processados salvos em processed/preprocessed_features.csv')