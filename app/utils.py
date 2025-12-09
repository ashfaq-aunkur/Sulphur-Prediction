import numpy as np

def apply_scaler_X(df_num, scaler_json):
    mean = np.array(scaler_json["mean"])
    scale = np.array(scaler_json["scale"])
    return (df_num - mean) / scale

def apply_encoder_X(df_cat, encoder_json):
    categories = encoder_json["categories"]
    encoded_arrays = []
    for i, col in enumerate(df_cat.columns):
        categories_i = categories[i]
        value = df_cat.iloc[0, i]
        one_hot = [1 if value == cat else 0 for cat in categories_i]
        encoded_arrays.append(one_hot)
    return np.array(encoded_arrays).flatten().reshape(1, -1)

        
