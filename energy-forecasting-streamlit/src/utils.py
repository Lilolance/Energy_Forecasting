import numpy as np

# custom MAPE function
def mape(actual, predicted):
    # Calculate Mean Absolute Percentage Error (MAPE)
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def get_train_val_test_split(data, train_size=0.7, val_size=0.15):
    # Split the data into train, validation, and test sets
    num_samples = len(data)
    train_end = int(train_size * num_samples)
    val_end = int((train_size + val_size) * num_samples)

    if isinstance(data, np.ndarray):
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
    else:
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]

    return train_data, val_data, test_data

