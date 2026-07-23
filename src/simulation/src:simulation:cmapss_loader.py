import pandas as pd
import numpy as np

# Define column names for C-MAPSS
COLUMN_NAMES = [
    'unit_number', 'time_in_cycles',
    'op_setting_1', 'op_setting_2', 'op_setting_3',
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4',
    'sensor_5', 'sensor_6', 'sensor_7', 'sensor_8',
    'sensor_9', 'sensor_10', 'sensor_11', 'sensor_12',
    'sensor_13', 'sensor_14', 'sensor_15', 'sensor_16',
    'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
    'sensor_21'
]

def load_cmapss(subset='FD001', data_type='train'):
    """
    Load C-MAPSS dataset for a specific subset and data type.
    
    Parameters:
    - subset: 'FD001', 'FD002', 'FD003', or 'FD004'
    - data_type: 'train' or 'test'
    
    Returns:
    - Pandas DataFrame with the data
    """
    filename = f"{data_type}_{subset}.txt"
    filepath = f"data/raw/{filename}"
    
    df = pd.read_csv(
        filepath,
        sep=r'\s+',  # C-MAPSS uses variable whitespace as delimiter
        header=None,
        names=COLUMN_NAMES
    )
    return df

def add_rul_labels(df):
    """
    Add Remaining Useful Life (RUL) as target column.
    RUL = max_time_for_unit - current_time
    For standard modeling, cap RUL at 125 cycles [citation:5].
    """
    max_cycles = df.groupby('unit_number')['time_in_cycles'].transform('max')
    df['rul'] = max_cycles - df['time_in_cycles']
    df['rul'] = df['rul'].clip(upper=125)
    return df

def drop_constant_sensors(df):
    """
    Remove sensors that don't change over time (no signal).
    For FD001: sensors 1, 5, 6, 10, 16, 18, 19 are constant [citation:5].
    """
    constant_sensors = ['sensor_1', 'sensor_5', 'sensor_6', 
                       'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
    return df.drop(columns=constant_sensors, errors='ignore')

# Quick test when script is run directly
if __name__ == "__main__":
    df_train = load_cmapss('FD001', 'train')
    df_train = add_rul_labels(df_train)
    df_train = drop_constant_sensors(df_train)
    
    print(f"Loaded {len(df_train)} rows from FD001 training set")
    print(f"Features: {list(df_train.columns)}")
    print("\nFirst 5 rows:")
    print(df_train.head())
    print("\nRUL distribution:")
    print(df_train['rul'].describe())