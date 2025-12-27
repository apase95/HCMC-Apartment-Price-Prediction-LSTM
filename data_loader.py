import pandas as pd
import os

class MarketDataLoader:
    def __init__(self):
        self.df = None

    def load_data_from_csv(self, filename):
        if not os.path.exists(filename):
            print(f"Không tìm thấy file: '{filename}'")
            return None

        try:
            self.df = pd.read_csv(filename)
            if 'Date' in self.df.columns:
                self.df['Date'] = pd.to_datetime(self.df['Date'])
                self.df.set_index('Date', inplace=True)
            else:
                self.df.iloc[:, 0] = pd.to_datetime(self.df.iloc[:, 0])
                self.df.set_index(self.df.columns[0], inplace=True)
            return self.df

        except Exception as e:
            print(f"Lỗi khi đọc file: {e}")
            return None