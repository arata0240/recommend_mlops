import os
import tempfile
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    base_dir = "/opt/ml/processing"
    #Read Data
    df = pd.read_csv(
        f"{base_dir}/input/movielens.csv"
    )


    # Encode the genre feature
    df['genre'] = df['genre'].apply(lambda x: ' '.join(eval(x)))  # Convert list strings to space-separated strings
    label_encoder = LabelEncoder()
    df['genre_encoded'] = label_encoder.fit_transform(df['genre'].astype(str))
    df['genre_encoded'] = df['genre_encoded'].fillna(0).astype(int)

    # Convert rating_order to float
    df['rating_order'] = df['rating_order'].astype(float)

    # Create the target variable based on the rating
    df['target'] = df['rating'].apply(lambda x: 1 if x > 3 else 0)

    # Prepare the feature set
    X = df[['genre_encoded', 'rating_order']]
    y = df['target']

    # Combine X and y for splitting
    data = np.concatenate((y.values.reshape(-1, 1), X.values), axis=1)
    np.random.shuffle(data)

    # Split the data into train, validation, and test sets
    train_size = int(0.7 * len(data))
    val_size = int(0.15 * len(data))
    train, validation, test = np.split(data, [train_size, train_size + val_size])

    # Convert the splits back into DataFrames
    train_df = pd.DataFrame(train, columns=['target', 'genre_encoded', 'rating_order'])
    validation_df = pd.DataFrame(validation, columns=['target', 'genre_encoded', 'rating_order'])
    test_df = pd.DataFrame(test, columns=['target', 'genre_encoded', 'rating_order'])

    # Save the datasets as CSV files
    train_df.to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    validation_df.to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    test_df.to_csv(f"{base_dir}/test/test.csv", header=False, index=False)