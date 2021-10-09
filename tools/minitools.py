import numpy as np
import pandas as pd


def convert_DF_dateTime(inputdf):
    """[Covert unisprot csv records datatime]

    Args:
        inputdf ([DataFrame]): [input dataFrame]

    Returns:
        [DataFrame]: [converted DataFrame]
    """
    inputdf.date_integraged = pd.to_datetime(inputdf['date_integraged'])
    inputdf.date_sequence_update = pd.to_datetime(inputdf['date_sequence_update'])
    inputdf.date_annotation_update = pd.to_datetime(inputdf['date_annotation_update'])
    inputdf = inputdf.sort_values(by='date_integraged', ascending=True)
    inputdf.reset_index(drop=True, inplace=True)
    return inputdf