
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

def load_data(filepath):
    df = pd.read_csv(filepath)
    logger.info(f"資料載入完成，共 {len(df)} 筆")
    return df

def standardize_features(df, feature_cols):
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    logger.info(f"特徵標準化完成，特徵數量: {len(feature_cols)}")
    return df, scaler

def prepare_time_series(df, features, sequence_length):
    sequences = []
    for i in range(len(df) - sequence_length + 1):
        seq = df[features].iloc[i:i + sequence_length].values
        sequences.append(seq)
    logger.info(f"時間序列資料準備完成，總序列數: {len(sequences)}")
    return sequences
