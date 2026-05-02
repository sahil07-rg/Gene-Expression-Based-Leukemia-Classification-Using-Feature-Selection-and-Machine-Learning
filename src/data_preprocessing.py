import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_data(train_path, test_path, label_path):
    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    labels_df = pd.read_csv(label_path)

    # Remove unnecessary columns
    train_df = train_df.drop(columns=['Gene Description'], errors='ignore')
    test_df = test_df.drop(columns=['Gene Description'], errors='ignore')

    # Set gene accession number as index
    train_df = train_df.set_index('Gene Accession Number')
    test_df = test_df.set_index('Gene Accession Number')

    # Remove call columns
    train_df = train_df.loc[:, ~train_df.columns.str.contains('call')]
    test_df = test_df.loc[:, ~test_df.columns.str.contains('call')]

    # Transpose
    train_df = train_df.T
    test_df = test_df.T

    # Combine datasets
    full_df = pd.concat([train_df, test_df], axis=0)

    # Encode labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels_df['cancer'])

    return full_df, y
