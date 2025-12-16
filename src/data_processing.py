import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df

def create_aggregates(df):
    agg_df = df.groupby('CustomerId').agg(
        total_amount=('Amount', 'sum'),
        avg_amount=('Amount', 'mean'),
        transaction_count=('TransactionId', 'count'),
        std_amount=('Amount', 'std')
    ).reset_index()
    return agg_df

def extract_time_features(df):
    df['transaction_hour'] = df['TransactionStartTime'].dt.hour
    df['transaction_day'] = df['TransactionStartTime'].dt.day
    df['transaction_month'] = df['TransactionStartTime'].dt.month
    df['transaction_year'] = df['TransactionStartTime'].dt.year
   
    time_agg = df.groupby('CustomerId').agg(
        avg_hour=('transaction_hour', 'mean'),
        avg_day=('transaction_day', 'mean')
    ).reset_index()
    return time_agg

def preprocess_pipeline():
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, ['total_amount', 'avg_amount', 'transaction_count', 'std_amount', 'avg_hour', 'avg_day']),
        ])
    return preprocessor

def process_data(input_path, output_path):
    df = load_data(input_path)
    
    agg_df = create_aggregates(df)
    time_df = extract_time_features(df)

    processed_df = agg_df.merge(time_df, on='CustomerId')
    processed_df.to_csv(output_path, index=False)
def preprocess_pipeline():
    numeric_features = [
        'total_amount',
        'avg_amount',
        'transaction_count',
        'std_amount',
        'avg_hour',
        'avg_day'
    ]

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numeric_features)
        ]
    )

    return preprocessor
    

if __name__ == "__main__":
    process_data('data/raw/data.csv', 'data/processed/processed.csv')