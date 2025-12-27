import numpy as np
import seaborn as sns
import tensorflow as tf
from data_loader import MarketDataLoader
from lstm_model import RealEstateLSTM

sns.set_style('whitegrid')
np.random.seed(2025)
tf.random.set_seed(2025)

if __name__ == "__main__":
    loader = MarketDataLoader()
    csv_file_path = "Dataset_GCC_HCM_2015_2025.csv"
    df_market = loader.load_data_from_csv(csv_file_path)

    if df_market is not None:
        ai = RealEstateLSTM(look_back=12)
        ai.step1_load_and_preprocess(df_market, test_months=24)
        ai.step2_build_model()
        ai.step3_train(epochs=50, batch_size=16)
        ai.step4_evaluate_and_visualize()
        ai.step5_predict_future()
    else:
        print("Thiếu dữ liệu.")