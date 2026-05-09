# third parties imports
import pandas as pd
# surprise
from surprise import Dataset, Reader
# export
from datetime import datetime

# local imports
from constants import Constant as C


def load_ratings(surprise_format=False):
    df_ratings = pd.read_csv(C.EVIDENCE_PATH / C.RATINGS_FILENAME)
    
    if surprise_format:
        reader = Reader(rating_scale=C.RATINGS_SCALE)
        dataset = Dataset.load_from_df(
            df_ratings[[C.USER_ID_COL, C.ITEM_ID_COL, C.RATING_COL]], 
            reader
        )
        return dataset
    else:
        return df_ratings


def load_items():
    df_items = pd.read_csv(C.CONTENT_PATH / C.ITEMS_FILENAME)
    df_items = df_items.set_index(C.ITEM_ID_COL)
    return df_items


def export_evaluation_report(df):
    
    C.EVALUATION_PATH.mkdir(parents=True, exist_ok=True)
    
    # The name of the report is versioned using today's date
    today_date = datetime.now().strftime("%Y_%m_%d")
    filename = f"{today_date}.csv"
    
    # Define the full destination path
    export_path = C.EVALUATION_PATH / filename
    
    #Save the DataFrame as a CSV
    df.to_csv(export_path, index=True)