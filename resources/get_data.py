import pandas as pd
from sklearn.model_selection import train_test_split


def get_data():
    df = pd.read_csv("data/train.csv")[:10]
    df["TEXT"] = df["TITLE"] + df["ABSTRACT"]

    label_columns = df.columns.tolist()[3:-1]
    print(df[label_columns].sum().sort_values())

    test_df, train_df = train_test_split(df, test_size=0.8, random_state=42)
    test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=42)

    return train_df, val_df, test_df
