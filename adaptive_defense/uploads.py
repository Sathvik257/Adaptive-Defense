"""Bounded parsing for hosted CSV analysis; no uploads are persisted."""
import pandas as pd

MAX_BYTES = 10 * 1024 * 1024
MAX_ROWS = 10000
MAX_COLUMNS = 256

def read_upload(file):
    file.seek(0,2)
    size=file.tell()
    file.seek(0)
    if size>MAX_BYTES:
        raise ValueError("Please use a CSV smaller than 10 MB.")
    try:
        frame=pd.read_csv(file,nrows=MAX_ROWS+1)
    except (UnicodeError,pd.errors.ParserError,pd.errors.EmptyDataError) as exc:
        raise ValueError("Please upload a valid UTF-8 CSV with a header row.") from exc
    if len(frame)>MAX_ROWS or len(frame.columns)>MAX_COLUMNS:
        raise ValueError("CSV limit: 10,000 rows and 256 columns.")
    if frame.empty:
        raise ValueError("The CSV contains no traffic records.")
    return frame
