import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações
dir_results = 'results'
dir_plots = 'plots'
s=os.makedirs(dir_results, exist_ok=True)
os.makedirs(dir_plots, exist_ok=True)

# 1) Carregar dados brutos de tráfego
# Usamos o CSV completo para manter timestamp e volume
df = pd.read_csv('processed/traffic_full.csv')
df['date_time'] = pd.to_datetime(df['date_time'])
df['hour'] = df['date_time'].dt.hour

def split_train_test(df, train_frac=0.7):
    split_time = df['date_time'].quantile(train_frac)
    train = df[df['date_time'] <= split_time]
    test  = df[df['date_time'] >  split_time]
    return train, test

train_df, test_df = split_train_test(df)

# 2) Calcular baseline por hora (média e desvio)
hour_stats = train_df.groupby('hour')['traffic_volume'] \
    .agg(['mean','std']).reset_index().rename(columns={'mean':'hour_mean','std':'hour_std'})

# 3) Identificar anomalias no teste
# Mesclar stats no test
test_df = test_df.merge(hour_stats, how='left', on='hour')
# Definir threshold k sigma (k=2)
test_df['threshold'] = test_df['hour_mean'] + 2 * test_df['hour_std']
# Flag de anomalia
anoms = test_df[test_df['traffic_volume'] > test_df['threshold']].copy()

# 4) Salvar resultados
train_df.to_csv(os.path.join(dir_results,'hourly_baseline_train.csv'), index=False)
anoms.to_csv(os.path.join(dir_results,'anomalies_by_hour_threshold.csv'), index=False)
print(f"Anomalias detectadas: {len(anoms)} de {len(test_df)} pontos no teste")

# 5) Gráfico: distribuição de anomalias por hora do dia
hour_counts = anoms['hour'].value_counts().sort_index()
plt.figure(figsize=(8,4))
sns.barplot(x=hour_counts.index, y=hour_counts.values)
plt.title('Anomalias por Hora do Dia (Threshold por Hora)')
plt.xlabel('Hora')
plt.ylabel('Número de Anomalias')
plt.tight_layout()
plt.savefig(os.path.join(dir_plots,'anomalies_by_hour_threshold.png'))
plt.close()

# 6) Timeline das anomalias
plt.figure(figsize=(12,4))
plt.plot(test_df['date_time'], test_df['traffic_volume'], alpha=0.3, label='Traffic')
plt.scatter(anoms['date_time'], anoms['traffic_volume'], color='red', s=10, label='Anomalias')
plt.title('Anomalias pelo Threshold por Hora - Timeline')
plt.xlabel('Date Time')
plt.ylabel('Volume')
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(dir_plots,'anomalies_timeline_hour_threshold.png'))
plt.close()

print('Análises por hora concluídas.')
