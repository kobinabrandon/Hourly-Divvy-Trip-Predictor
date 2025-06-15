import os 
from pathlib import Path 


PARENT_DIR = Path("_file_").parent.resolve()

IMAGES_DIR = PARENT_DIR.joinpath("images")
DATA_DIR = PARENT_DIR.joinpath("data")

RAW_DATA_DIR = DATA_DIR.joinpath("raw")

MODELS_DIR = PARENT_DIR.joinpath("models")
LOCAL_SAVE_DIR = MODELS_DIR.joinpath("local_saves")
COMET_SAVE_DIR = MODELS_DIR.joinpath("comet_downloads")

CLEANED_DATA = DATA_DIR.joinpath("cleaned")
TRANSFORMED_DATA = DATA_DIR.joinpath("transformed")
GEOGRAPHICAL_DATA = DATA_DIR.joinpath("geographical")

ROUNDING_INDEXER = GEOGRAPHICAL_DATA.joinpath("rounding_indexer")
MIXED_INDEXER = GEOGRAPHICAL_DATA.joinpath("mixed_indexer")

TIME_SERIES_DATA = TRANSFORMED_DATA.joinpath("time_series")
TRAINING_DATA = TRANSFORMED_DATA.joinpath("training_data")
INFERENCE_DATA = TRANSFORMED_DATA.joinpath("inference")

START_TS_PATH = TIME_SERIES_DATA.joinpath("start_ts.parquet")
END_TS_PATH = TIME_SERIES_DATA.joinpath("end_ts.parquet")


def make_fundamental_paths() -> None:
    for path in [
        DATA_DIR, CLEANED_DATA, RAW_DATA_DIR, GEOGRAPHICAL_DATA, TRANSFORMED_DATA, TIME_SERIES_DATA, 
        IMAGES_DIR, TRAINING_DATA, INFERENCE_DATA, MODELS_DIR, LOCAL_SAVE_DIR, COMET_SAVE_DIR, ROUNDING_INDEXER,
        MIXED_INDEXER
    ]: 
        if not Path(path).exists():
            os.mkdir(path)

