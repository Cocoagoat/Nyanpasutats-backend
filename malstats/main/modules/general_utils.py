from __future__ import print_function

import datetime
import http.client
import traceback
from http import HTTPStatus
import numpy as np
import requests
import time
import pickle
import logging
import pandas as pd
from enum import Enum
from functools import wraps
import re
from polars.exceptions import SchemaFieldNotFoundError
import main.modules.GenerateAccessToken as AccessToken
import polars as pl
import csv
import sys
from requests.exceptions import SSLError, JSONDecodeError, ChunkedEncodingError
from urllib3.exceptions import ProtocolError, ConnectionError
import xml.etree.ElementTree as ET
try:
    import thread
except ImportError:
    import _thread as thread
from django.core.cache import cache
from polars.exceptions import ColumnNotFoundError
from main.modules.filenames import *
from django.conf import settings
from main.modules.Errors import UserListPrivateError, UserDoesNotExistError, UserListFetchError

logger = logging.getLogger("nyanpasutats")


class ErrorCauses(Enum):
    RESOURCE_LOCKED = "RESOURCE_LOCKED"
    HEADERS_EXPIRED = "HEADERS_EXPIRED"
    TOO_MANY_REQUESTS = "TOO_MANY_REQUESTS"
    SITE_DOWN = "SITE_DOWN"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    NOT_FOUND = "NOT_FOUND"
    BAD_REQUEST = "BAD_REQUEST"


class Sleep:
    def __init__(self, t):
        self.time = t

    LONG_SLEEP = 300
    AUTH_SLEEP = 30 # Time given to authenticate in case of 401 error
    MEDIUM_SLEEP = 5
    SHORT_SLEEP = 0.5

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, t):
        self._time = t


class DjangoUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "modules.GeneralData":
            module = "main.modules.GeneralData"
        return super().find_class(module, name)


def timeit(method):
    """A decorator that can be added to any function to count the amount of time it ran for."""

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print(f'{method.__name__} ran for {(te - ts)} seconds')
        return result

    return timed


def time_at_current_point(init_time, message=None):
    curr_time = time.perf_counter()
    print(f"Elapsed time : {curr_time-init_time}. {message}")


def terminate_program():
    sys.exit(1)


def get_headers():
    headers = {}
    # with open('Authorization.txt', "r") as f:
    with open(auth_filename) as f:
        headers_list = f.read().splitlines()
        if headers_list:
            headers['Authorization'] = headers_list[0]
            headers['refresh_token'] = headers_list[1]
        headers['content-type'] = 'application/x-www-form-urlencoded'
    return headers


def fetch_new_headers():
    access_token = AccessToken.get_access_token()
    with open(auth_filename, "w") as f:
        f.write(f'Bearer {access_token["access_token"]}')
        f.write("\n")
        f.write(f'{access_token["refresh_token"]}')
    time.sleep(3)
    return get_headers()


try:
    headers = get_headers()
except FileNotFoundError:
    print("Warning : Authorization file not found. (This is fine if utils"
          "are not used for MyAnimeList)")
    headers = []


def add_to_queue(q, item):
    if q is not None:
        q.put(item)


def list_to_uint8_array(lst):
    arr = np.array(lst, dtype=np.float32)  # First, we convert to float to handle None values
    arr = np.nan_to_num(arr, nan=0).astype(np.uint8)  # Now we can replace nan with 0 and convert to uint8
    return arr


def sort_dict_by_values(d, reverse=True):
    return {k: v for k, v in sorted(d.items(), key=lambda x: x[1], reverse=reverse)}


def load_pickled_file(filename):
    with open(filename, 'rb') as f:
        try:
            pickled_file = pickle.load(f)
        except ModuleNotFoundError:
            f.seek(0)
            pickled_file = DjangoUnpickler(f).load()
    return pickled_file


def save_pickled_file(filename,obj):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)


def count_calls(func):
    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        print(f'Call {wrapped.calls} of {func.__name__!r}')
        return func(*args, **kwargs)
    wrapped.calls = 0
    return wrapped


def split_list_interval(input_list, n_parts):
    return [input_list[i::n_parts] for i in range(n_parts)]


def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    anime_list = []

    for anime in root.findall('anime'):
        anime_data = {}
        for child in anime:
            anime_data[child.tag] = child.text.strip() if child.text else None
        anime_list.append(anime_data)

    return anime_list


def get_data(url, site=None):
    new_headers_fetched = False
    for retry_count in range(100):
        try:
            time.sleep(1.2)
            headers = get_headers()
            response = requests.get(url, headers=headers, timeout=15)
            # logger.info(response)
            if response.status_code == HTTPStatus.OK:
                try:
                    return response.json()
                except JSONDecodeError:
                    return response
            elif response.status_code == HTTPStatus.UNAUTHORIZED.value:
                if not new_headers_fetched:
                    headers = fetch_new_headers()
                    new_headers_fetched = True
                    continue
                else:
                    logger.critical(f"Response for url {url} was {response.status_code}, and acquisition of"
                                    f"new headers failed to remedy the problem.")
                    raise UserListFetchError
            elif response.status_code == HTTPStatus.FORBIDDEN.value:
                logger.error(f"Response for url {url} was {response.status_code}, resource is private")
                raise UserListPrivateError
            elif (response.status_code == HTTPStatus.NOT_FOUND.value
            ) or response.status_code == HTTPStatus.METHOD_NOT_ALLOWED.value:
                raise UserDoesNotExistError
            else:
                logger.error(f"Response for url {url} was {response.status_code}, cause is unknown")
                continue
        except (SSLError, ConnectionError, requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError, http.client.RemoteDisconnected,
                ProtocolError, ChunkedEncodingError, ConnectionResetError) as e:
            # Sometimes MAL throws a weird SSL error, retrying fixes it
            if retry_count < 10:
                time.sleep(Sleep.SHORT_SLEEP)
            elif retry_count < 50:
                time.sleep(Sleep.LONG_SLEEP)
            else:
                time.sleep(Sleep.LONG_SLEEP * 6)
            print("Unable to connect, retrying")
            logger.warning("Unable to connect, retrying")
            continue

    logger.critical("FATAL ERROR - Unable to connect to server for an extended period of time.")
    raise ConnectionRefusedError("FATAL ERROR - Unable to connect to server for an extended period of time.")


def save_list_to_csv(lst, filename):
    try:
        with open(filename, "w", encoding='utf-8', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(lst)
    except PermissionError:
        pass
        # logger.error("Unable to save list. Please close the CSV file in Excel")
        # Non-critical function, we can proceed without saving the list as CSV (though
        # using continue_filling_database will be a problem if the program crashes before
        # the list is saved),


def filter_rows_by_column_values(df,col_name, bad_values_list):
    """Saved for future use"""
    filtered_df = df.filter(
        pl.col("Username").apply(lambda x: x not in bad_values_list, return_dtype=pl.Boolean))


def reindex_df(df : pl.DataFrame):
    """Saved for future use"""
    try:
        df.drop('Index')
    except SchemaFieldNotFoundError:
        pass
    new_dict = {'Index' : [i for i in range(df.shape[0])]}
    new_row = pl.DataFrame(new_dict)
    df = pl.concat([new_row,df],how="horizontal")
    return df


def find_duplicates(lst):
    sorted_lst = sorted(lst)
    index_duplicate_list = []
    for i in range(len(lst)-1):
        if sorted_lst[i]==sorted_lst[i+1]:
            index_duplicate_list.append((i, lst[i]))
    return index_duplicate_list


def synchronize_dfs(df1, df2, cols1=None, cols2=None):
    # Add all columns in db2 but not in db1 to db1
    if cols1 and cols2:
        missing_cols = [x for x in cols2 if x not in cols1]
    else:
        missing_cols = [x for x in df2.columns if x not in df1.columns]

    for col_name in missing_cols:
        df1 = df1.with_columns(
            pl.Series(col_name, [None] * len(df1), dtype=pl.UInt8))

    return df1


def concat_to_existing_dict(dict1_filename,dict2,concat_type):
    """Concatenates two dictionaries - merging them if they have different keys or extending each list
    if they are dictionaries of lists"""
    try:
        dict1 = load_pickled_file(dict1_filename)
    except FileNotFoundError:
        dict1 = {}

    if concat_type == "merge" or not dict1.keys():
        dict1 = dict1 | dict2
    elif concat_type == "append":

        for key in dict1.keys():
            try:
                dict1[key] = dict1[key] + dict2[key]
            except KeyError:
                continue
    else:
        logger.error(f"Bad concatenation type {concat_type} when trying to concatenate to {dict1_filename}")
        raise ValueError

    save_pickled_file(dict1_filename, dict1)


def remove_zero_columns(df : pl.DataFrame):
    zero_columns = [col for col in df.columns if df[col].sum() == 0]
    for col in zero_columns:
        if col.endswith("Affinity"):
            # If say, "Crime Affinity" is full of zeros,
            # we want to remove it and the "Crime" column.
            try:
                df.drop_in_place(col)
                df.drop_in_place(" ".join(col.split()[:-1]))
            except ColumnNotFoundError:
                continue
        else:
            try:
                df.drop_in_place(col)
                df.drop_in_place(col + " Affinity")
            except ColumnNotFoundError:
                continue


def shuffle_df(df: pd.DataFrame):
    return df.sample(frac=1).reset_index(drop=True)


def handle_nans(df):
    if df.isna().any().any():
        has_nans_per_column = df.isna().any()
        for col in df.columns:
            if has_nans_per_column[col]:
                df[col].fillna(0, inplace=True)
    return df


def split_list_interval(input_list, n_parts):
    return [input_list[i::n_parts] for i in range(n_parts)]


def print_attributes(obj):
    for attr in dir(obj):
        if not callable(attr) and attr.startswith("_") and not attr.startswith("__"):
            value = getattr(obj, attr)
            print(f"{attr} : {str(value)[:50]}")


def snake_to_camel(snake_str):
    # Function to convert snake_case string to CamelCase
    components = snake_str.split('_')
    return components[0].capitalize() + ''.join(x.title() for x in components[1:])


def camel_to_snake(name):
    # Add an underscore before each capital letter (except the first one) and convert to lower case
    snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
    return snake_case


def convert_keys_to_camel_case(data):
    if isinstance(data, dict):
        # Recursively apply to dictionary keys
        return {snake_to_camel(k): convert_keys_to_camel_case(v) for k, v in data.items()}
    elif isinstance(data, list):
        # Recursively apply to elements in lists
        return [convert_keys_to_camel_case(item) for item in data]
    else:
        # Base case: when data is neither a dict nor a list,
        # just return the data itself
        return data


def add_suffix_to_filename(full_filename: Path, suffix: str) -> Path:
    """Adds a suffix of the type "-SUFFIX" to a filename.
    Example: suffix = U, filename = AnimeDB.parquet, output = AnimeDB-U.parquet"""

    if not isinstance(full_filename, Path):
        raise TypeError("full_filename must be a pathlib.Path object")

    file_stem = full_filename.stem
    file_ext = full_filename.suffix

    new_filename = f"{file_stem}-{suffix}{file_ext}"
    return full_filename.with_name(new_filename)


def is_redis_cache():
    # Access the backend of the default cache
    backend = settings.CACHES['default']['BACKEND']

    # Check if the backend is set to django_redis.cache.RedisCache
    return 'django_redis.cache.RedisCache' in backend


def redis_cache_wrapper(timeout):
    def decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            cache_key = f"{function.__name__}_{'_'.join(str(arg) for arg in args)}_{'_'.join(f'{key}_{value}' for key, value in kwargs.items())}"

            result = cache.get(cache_key)
            if result is not None:
                return result

            result = function(*args, **kwargs)
            if not (isinstance(result, dict) and 'error' in result):
                cache.set(cache_key, result, timeout)

            return result
        return wrapped
    return decorator


def rate_limit(rate_lim=1, cache_key=None):
    def decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            print("Entering rate limiter")
            nonlocal cache_key
            if not cache_key:
                if "site" in kwargs.keys():
                    cache_key = f"{kwargs['site']}_rate_limit"
                else:
                    raise ValueError("If the decorator doesn't have a cache_key,"
                                     "the wrapped function must provide a site")

            print("Cache key is", cache_key)
            first_time = True
            while not can_make_api_call(rate_lim, cache_key, first_time):
                first_time = False
                time.sleep(0.1)
            print("Entering function")
            print(args, kwargs)
            result = function(*args, **kwargs)
            return result
        return wrapped
    return decorator


def can_make_api_call(rate_lim, cache_key, first_time):

    current_time = int(time.time())
    last_api_call_time = cache.get(cache_key, 0)

    if last_api_call_time is None or (current_time - int(last_api_call_time)) >= rate_lim:
        # Update the last API call time and proceed
        cache.set(cache_key, current_time)
        if not first_time:
            logger.info(f"Waiting for rate limit. {rate_lim - (current_time - last_api_call_time)} seconds left.")
        return True
    else:
        # Wait if we're within the rate limit window
        return False


def convert_to_timestamp(date_str, time_str):
    # Parse the input date and time strings
    date_format = "%b %d"  # Format for month (3 letters) and day
    time_format = "%I:%M %p"  # Format for 12-hour clock with AM/PM

    # Get the current year to form a complete date string
    current_year = datetime.datetime.now().year

    # Combine date and year to form a complete date string
    full_date_str = f"{date_str} {current_year}"
    full_date_format = f"{date_format} %Y"

    # Parse the complete date and time strings into datetime objects
    date_obj = datetime.datetime.strptime(full_date_str, full_date_format)
    time_obj = datetime.datetime.strptime(time_str, time_format)

    # Combine date and time into a single datetime object
    combined_datetime = date_obj.replace(hour=time_obj.hour, minute=time_obj.minute, second=0, microsecond=0)

    return combined_datetime.timestamp()


def determine_queue_cache_key(source_name, source_type="task"):
    """Determines the cache key used to keep track of the request queue of each task/view.
     (tasks_in_queue_seasonal for the seasonal task, etc)"""
    if source_name.endswith('seasonal_stats_task') or 'seasonal' in source_name and source_type == "view":
        queue_cache_key = 'tasks_in_queue_seasonal'
    elif source_name.endswith('recs_task') or 'recs' in source_name and source_type == "view":
        queue_cache_key = 'tasks_in_queue_recs'
    elif source_name.endswith('affs_task') or 'affs' in source_name and source_type == "view":
        queue_cache_key = 'tasks_in_queue_affs'
    else:
        raise ValueError(f"Unknown task/view {source_name}. "
                         f"Make sure the names of the tasks have not been changed without"
                         "applying these changes here.")
    return queue_cache_key


def basic_try_except(func):
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"An unexpected error has occurred during the daily update. {str(e)}")
            return None
    return inner


def log_exception_error(logger, log_message):
    full_stack_trace = traceback.format_exc()
    logger.error(log_message)
    logger.error(full_stack_trace)




