import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações
dir_results = 'results'
dir_plots = 'plots'
window_sizes = [7, 14, 30]  # janelas em dias
os.makedirs(dir_results, exist_ok=True)
os.makedirs(dir_plots, exist_ok=True)

# 1) Carregar dados brutos
df = pd.read_csv('processed/traffic_full.csv')
df['date_time'] = pd.to_datetime(df['date_time'])
df.sort_values('date_time', inplace=True)
df.reset_index(drop=True, inplace=True)
# Extrair hora
df['hour'] = df['date_time'].dt.hour

# 2) Para cada janela, calcular threshold móvel e gerar resultados
for window in window_sizes:
    df_copy = df.copy()
    days = window  # número de dias para rolling
    
    # Rolling: precisamos de tantas observações quanto dias por hora
    df_copy['rolling_mean'] = df_copy.groupby('hour')['traffic_volume'].transform(lambda x: x.shift().rolling(window=days, min_periods=3).mean())
    df_copy['rolling_std'] = df_copy.groupby('hour')['traffic_volume'].transform(lambda x: x.shift().rolling(window=days, min_periods=3).std())

    # Threshold e flag
    df_copy['threshold'] = df_copy['rolling_mean'] + 2 * df_copy['rolling_std']
    df_copy['is_anomaly'] = df_copy['traffic_volume'] > df_copy['threshold']
    anoms = df_copy[df_copy['is_anomaly']].copy()
    print(f"Window {window} dias: {len(anoms)} anomalias de {len(df_copy)} pontos")

    # 3) Salvar CSV de anomalias
    fname = f'rolling_{window}d_anomalies.csv'
    anoms.to_csv(os.path.join(dir_results, fname), index=False)

    # 4) Resumo por hora
    summary = df_copy.groupby('hour').agg(
    total=('traffic_volume','size'),anomalies=('is_anomaly','sum')).reset_index()
    summary['anomaly_rate'] = summary['anomalies'] / summary['total']
    sname = f'rolling_{window}d_summary.csv'
    summary.to_csv(os.path.join(dir_results, sname), index=False)

    # 5) Plot taxa de anomalias por hora
    plt.figure(figsize=(10,4))
    sns.lineplot(x='hour', y='anomaly_rate', data=summary, marker='o')
    plt.title(f'Taxa de Anomalias por Hora (Rolling {window} dias)')
    plt.xlabel('Hora')
    plt.ylabel('Taxa de Anomalias')
    plt.tight_layout()
    pname = f'rolling_{window}d_rate_by_hour.png'
    plt.savefig(os.path.join(dir_plots, pname))
    plt.close()

    # 6) Timeline completo
    plt.figure(figsize=(12,4))
    plt.plot(df_copy['date_time'], df_copy['traffic_volume'], alpha=0.3)
    plt.scatter(anoms['date_time'], anoms['traffic_volume'], color='red', s=8)
    plt.title(f'Anomalias (Rolling {window} dias)')
    plt.xlabel('Date Time')
    plt.ylabel('Volume')
    plt.tight_layout()
    tname = f'rolling_{window}d_timeline.png'
    plt.savefig(os.path.join(dir_plots, tname))
    plt.close()

print('Análises para janelas de 14 e 30 dias concluídas.')
