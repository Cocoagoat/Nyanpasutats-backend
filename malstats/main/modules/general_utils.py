from __future__ import print_function

import http.client
from http import HTTPStatus
import numpy as np
import requests
import time
import pickle
import pandas as pd
from enum import Enum
from functools import wraps
import re

from django.db.models import Case, When

from .GlobalValues import CACHE_TIMEOUT
from polars.exceptions import SchemaFieldNotFoundError
import main.modules.GenerateAccessToken as AccessToken
# try:
#     import GenerateAccessToken as AccessToken
# except (ImportError, ModuleNotFoundError) as ex:
#     print(ex)
#     print("Warning : GenerateAccessToken not found")
import logging
import polars as pl
import csv
import sys
from requests.exceptions import SSLError, JSONDecodeError
from urllib3.exceptions import ProtocolError, ConnectionError
import multiprocessing as mp
import xml.etree.ElementTree as ET
try:
    import thread
except ImportError:
    import _thread as thread
from colorama import Fore
from django.core.cache import cache
from polars.exceptions import ColumnNotFoundError
from main.modules.filenames import *
from django.conf import settings
from main.modules.Errors import UserListPrivateError, UserDoesNotExistError, UserListFetchError
# from .AffinityDB import GeneralData
# from . import AffinityDB


class ErrorCauses(Enum):
    RESOURCE_LOCKED = "RESOURCE_LOCKED"
    HEADERS_EXPIRED = "HEADERS_EXPIRED"
    TOO_MANY_REQUESTS = "TOO_MANY_REQUESTS"
    SITE_DOWN = "SITE_DOWN"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    NOT_FOUND = "NOT_FOUND"
    BAD_REQUEST = "BAD_REQUEST"


class UnknownServerException(Exception):
    pass


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


# def call_function_through_process(func, *args):
#     """The very painful result of the MAL API being a forsaken prehistoric pile of junk. I
#     am not experienced enough to know what the hell is wrong with it, but I have DEFINITELY
#     spent enough time to know that something is.\n
#     This function creates a separate process to call another function in, and works with queues
#     to return the result. It is needed for the sole purpose of a forced timeout during a call to
#     the MAL API. The timeout is required because during a call, there is about a 1/500 chance of
#     the GET request to get completely stuck. It will never timeout by itself, nor will it return
#     any error. The program will just endlessly wait."""
#     q = mp.Queue()
#     p = mp.Process(target=func, args=(*args, q))
#     p.start()
#     unauthorized = False
#     time_start = time.time()
#     timeout = 15
#     timeout_flag = True
#     value = None
#     while time.time() - time_start <= timeout:
#         if p.is_alive():
#             if not q.empty():
#                 value = q.get()
#                 # print(f"q is not empty, value is {value}")
#                 # print(type(value))
#                 if isinstance(value, Sleep):
#                     time_start = time_start + value.time
#                     print(f"Command to sleep {value.time} seconds"
#                           f"received from child process. Commencing sleep")  #
#                     time.sleep(value.time)
#                 if value == "UNAUTHORIZED":
#                     print("Using the normal get_search_results for headers")
#                     return get_search_results(*args)
#             else:
#                 time.sleep(Sleep.SHORT_SLEEP/5)
#         else:
#             print("Timeout flag is false, breaking loop")
#             timeout_flag = False
#             break
#     if timeout_flag:
#         print(Fore.LIGHTWHITE_EX + "Process is still alive, terminating process")
#         logger.debug("TIMEOUT ERROR - Process is still alive, terminating process")
#         p.terminate()
#     else:
#         p.join()
#         print(Fore.LIGHTWHITE_EX + "Process successfully finished on time")
#         logger.debug("Process successfully finished on time")
#     # print(f"returned value is {value}")
#     return value


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
    headers=[]


def add_to_queue(q, item):
    if q is not None:
        q.put(item)


def list_to_uint8_array(lst):
    arr = np.array(lst, dtype=np.float32)  # First, we convert to float to handle None values
    arr = np.nan_to_num(arr, nan=0).astype(np.uint8)  # Now we can replace nan with 0 and convert to uint8
    return arr


def sort_dict_by_values(d, reverse=True):
    return {k: v for k, v in sorted(d.items(), key=lambda x: x[1], reverse=reverse)}


def determine_unauthorized_cause(q=None):
    """ This function uses a URL that's known to be available to determine the exact
        cause of an unauthorized/forbidden (401/403) error. It is necessary because during MAL's
        authorization process, you can sometimes get a 403 error from every resource
        despite getting authorized + the MAL API returns a 403 error instead of a 429
        (Too Many Requests) error, so the causes of these two errors may not be what
        they are supposed to be."""

    global headers
    dummy_url = f'https://api.myanimelist.net/v2/users/BaronBrixius/animelist'
    # The dummy_url link is known to be an available resource - requesting it should
    # return code 200, unless there's a server-side or client-side issue.

    dummy_response = requests.get(dummy_url, headers=headers)
    print(f'{dummy_response} after trying dummy url request')

    # if response.status_code == 500:
    #     print("Site temporarily down, stopping program until it's back up")
    #     return ErrorCauses.SITE_DOWN

    if dummy_response.status_code == 200:
        # If the request succeeded, the problem was with the resource
        # we tried to access.
        print('Resource was locked, proceeding to next anime list')
        # logger.debug('Resource was locked, proceeding to next anime list')
        return ErrorCauses.RESOURCE_LOCKED

    if dummy_response.status_code == 401 or dummy_response.status_code == 403:
        # If we got 401/403 again, it's a client-side authorization issue. We need to
        # get a new access token for OAuth2 Authorization.
        print("Response is 401/403, the problem was not the url itself."
              " Testing if the access token has expired.")  #
        print(f'Headers while the problem occurred : \n {headers}')
        # print(f'Queue is {q}')
        # q.put(Sleep.AUTH_SLEEP)
        # time.sleep(Sleep.MEDIUM_SLEEP)
        if q is not None:
            # get_search_results(dummy_url)
            q.put("UNAUTHORIZED")
            time.sleep(Sleep.SHORT_SLEEP*3)
            # Give call_function_through_process time to get from queue
            return
        access_token = AccessToken.get_access_token()
        with open(auth_filename, "w") as f:
            f.write(f'Bearer {access_token["access_token"]}')
            f.write("\n")
            f.write(f'{access_token["refresh_token"]}')

        time.sleep(5)
        headers = get_headers()
        print(f'New headers: \n {headers}')
        response = requests.get(dummy_url, headers=headers)
        print(f'Trying the dummy request after requesting a new Bearer '
              f'Token, response is {response} \n')  #

        if response.status_code == 200:
            # If we got 200 after getting new headers, the problem was solved.
            print('Headers were expired, successfully acquired new headers.'
                  'Proceeding with same anime list')
            # logger.debug('Headers were expired, successfully acquired new headers.'
            #              'Proceeding with same anime list')
            return ErrorCauses.HEADERS_EXPIRED

        if response.status_code == 401 or response.status_code == 403:
            # If we got 401/403 even after receiving new headers,
            # the problem might be in MAL's "shadow rate limit",
            # which causes a 401/403 error rather than 429 Too Many Requests
            print("Response is 401/403, the problem was not in the headers."
                  "Testing for rate limit")  #
            # logger.debug("Response is 401/403, the problem was not in the headers."
            #              "Assuming rate limit issue and trying again")  #
            add_to_queue(q, Sleep(Sleep.LONG_SLEEP))
            time.sleep(Sleep.LONG_SLEEP)
            response = requests.get(dummy_url, headers=headers)

            if response.status_code == 200:
                # If we got 200 after sleeping for 10 minutes,
                # the problem was indeed the shadow rate limit.
                print('Too many requests were sent, issue resolved.'
                      ' Proceeding with same anime list')  #
                return ErrorCauses.TOO_MANY_REQUESTS
            return ErrorCauses.UNKNOWN_ERROR
    # 401/403 keeps getting returned even after acquiring headers and sleeping

    return ErrorCauses.UNKNOWN_ERROR
    # This line is normally unreachable, unless we get an error 403 from the original
    # request and then suddenly a different error from the dummy request.


def analyze_unauthorized_cause(unauthorized_cause, url, q=None):
    """ This function returns the appropriate value according to the cause
        of the unauthorized error in the original request.

        Parameters :

             unauthorized_cause - One of the values from the ErrorCauses enum.
             See comments in the function's text for more info on each case.

             url - Original request url

             q - The queue used in case get_search_results was called through
                 a separate process. It is necessary in this function in case
                 we want to sleep (and thus also need to tell the main process
                 to sleep, since this will run inside the child process)."""
    if unauthorized_cause == ErrorCauses.RESOURCE_LOCKED:
        # Problem is resource-specific (e.g user list is locked),
        # moving on to next resource since this one cannot be retrieved.
        return "RESOURCE_LOCKED"
    if unauthorized_cause == ErrorCauses.HEADERS_EXPIRED or ErrorCauses.TOO_MANY_REQUESTS:
        # Cause was determined to be expired headers/shadow rate limit,
        # retrying the same request again.
        response = get_search_results(url)
        return response
    # ----------------------------------------------------------------------------------#
    # 99.99% of the time, the above two cases are the only reasons we could get an
    # authorization error. The below case is handled for the sake of absolute safety
    # since the program runs for multiple days at a time.
    if unauthorized_cause == ErrorCauses.UNKNOWN_ERROR:
        time.sleep(Sleep.MEDIUM_SLEEP)
        unauthorized_cause = determine_unauthorized_cause(q)

        # Trying to determine unauthorized cause one more time. If it still returns
        # an unknown error, we move on to the next resource.
        if unauthorized_cause == ErrorCauses.UNKNOWN_ERROR:
            get_search_results.consequent_unknown_errors += 1
        else:
            return analyze_unauthorized_cause(unauthorized_cause, url, q)

            # If 10 resources in a row failed on authorization AND the dummy request
            # returned an unknown error after each time, we'll sleep for a long time
            # and see if that fixes the problem. If not, we terminate.
        if get_search_results.consequent_unknown_errors == 10:
            add_to_queue(q, Sleep(Sleep.LONG_SLEEP))  # Telling main process to sleep
            # as well
            time.sleep(Sleep.LONG_SLEEP * 6)
            unauthorized_cause = determine_unauthorized_cause(q)
            if unauthorized_cause == ErrorCauses.UNKNOWN_ERROR:
                # logger.error('An unknown server exception has occurred'
                #              ' over 10 times in a row, terminating program.')  #
                raise UnknownServerException
        return None
    print("No case matched, returning None")
    return None


def load_pickled_file(filename):
    test = __name__
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
    for _ in range(10):
        try:
            time.sleep(1)
            headers=get_headers()
            response = requests.get(url, headers=headers, timeout=15)
            print(response)
            if response.status_code == HTTPStatus.OK:
                return response.json()
            elif response.status_code == HTTPStatus.UNAUTHORIZED.value:
                headers = fetch_new_headers()
                continue
            elif response.status_code == HTTPStatus.FORBIDDEN.value:
                raise UserListPrivateError
            elif response.status_code == HTTPStatus.NOT_FOUND.value:
                raise UserDoesNotExistError
            else:
                print(response.status_code)
                continue
        except (SSLError, ConnectionError, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError, http.client.RemoteDisconnected, ProtocolError, ConnectionResetError) as e:
            # Sometimes MAL throws a weird SSL error, retrying fixes it
            time.sleep(Sleep.SHORT_SLEEP)
            print("Error, retrying connection")
            # logger.debug("Error, retrying connection")
            continue

    raise UserListFetchError("Unable to connect to server. Please try again later.")


def get_search_results(url, q=None):
    """ The main "get" function, tweaked to automatically handle various errors as the
    main program needs to run several days without stopping

        Parameters :

            url - the url from which to get resource
            q - In case this function was run inside a separate (child) process,
            this will be queue through which we pass the resource/other things
            to the main process.

        Return values :

            response - JSONified if possible, full response if not (for example,
            if the response is raw HTML).
            None - the function will return None in case it determines that the resource
            we are trying to get is unavailable.

        Errors :

            The function is not programmed to raise an error in any but the most
            severe of cases (see comments), since for our purposes skipping a resource is
            far preferred over stopping the program which makes 50,000+ requests over its
            runtime.
        """

    for _ in range(10):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            time.sleep(1)
            # print(response.status_code)
            # print(response.json())
        except (SSLError, ConnectionError, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError, http.client.RemoteDisconnected, ProtocolError, ConnectionResetError) as e:
            # Sometimes MAL throws a weird SSL error, retrying fixes it
            time.sleep(Sleep.SHORT_SLEEP)
            print("Error, retrying connection")
            # logger.debug("Error, retrying connection")
            continue
        break

    codes_to_check = [HTTPStatus.OK, HTTPStatus.NOT_FOUND, HTTPStatus.BAD_REQUEST,
                      HTTPStatus.INTERNAL_SERVER_ERROR, HTTPStatus.BAD_GATEWAY,
                      HTTPStatus.SERVICE_UNAVAILABLE, HTTPStatus.REQUEST_TIMEOUT]

    if response.status_code != HTTPStatus.OK:
        # logger.debug(response.status_code)
        if response.status_code == HTTPStatus.UNAUTHORIZED or response.status_code == HTTPStatus.FORBIDDEN:
            # Due to the way the MAL API works, the error 401/403 case is very complex
            # to handle if we want our program to keep running without any errors.
            unauthorized_cause = determine_unauthorized_cause(q)
            response_cause = analyze_unauthorized_cause(unauthorized_cause, url, q)
            if response_cause == "RESOURCE_LOCKED":
                print(response, 5)
                raise UserListPrivateError
                # return

        elif response.status_code == HTTPStatus.NOT_FOUND:  # If it's a list, user was probably deleted
                # test sleep queue here
            # logger.error("Resource does not exist, moving on to next resource")
            raise UserDoesNotExistError

        elif response.status_code == HTTPStatus.BAD_REQUEST:
            # This should never happen within the scope of the program
            # logger.error("There was a problem with the request itself,"
            #              "moving on to next resource")  #
            raise UserListFetchError("Unknown error fetching list")

        elif response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR\
                or response.status_code == HTTPStatus.BAD_GATEWAY\
                or response.status_code == HTTPStatus.SERVICE_UNAVAILABLE\
                or response.status_code == HTTPStatus.REQUEST_TIMEOUT:
            # If site is down, all we can do is retry every once in a while until
            # it works.
            # logger.error(f"Site is down. Retrying in {Sleep.LONG_SLEEP} seconds.")  #
            if q:
                add_to_queue(q, Sleep(Sleep.LONG_SLEEP))
            time.sleep(Sleep.LONG_SLEEP)
            response = get_search_results(url)

            # if response.status_code == HTTPStatus.SERVICE_UNAVAILABLE\
            #         or response.status_code == HTTPStatus.REQUEST_TIMEOUT:
            #     print(f"Service Unavailable/Requested timeout. Retrying in {Sleep.LONG_SLEEP} seconds.")
            #     logger.error(f"Service Unavailable/Requested timeout. Retrying in {Sleep.LONG_SLEEP} seconds.")
            #     print(q)
            #     if q:
            #         add_to_queue(q, Sleep(Sleep.LONG_SLEEP))
            #     time.sleep(Sleep.LONG_SLEEP)
            #     response=get_search_results(url)

        if not response.status_code or response.status_code not in codes_to_check:
            try:
                # logger.error(response.raise_for_status())
                print("Unknown API Error, trying to sleep")
                if q:
                    add_to_queue(q, Sleep(Sleep.LONG_SLEEP))
                time.sleep(Sleep.LONG_SLEEP)
                response = get_search_results(url)
            except requests.HTTPError as ex:
                # logger.error(ex)
                print("Unknown API Error, cannot continue")
                terminate_program()

    get_search_results.consequent_unknown_errors = 0  # If we got to this line,
    # error count is reset. The count is used by analyze_unauthorized_cause only.

    try:
        if q is not None:
            # If we're calling the function through a separate process, we want
            # to pass it our return value through the queue.
            # print("Putting json into queue")
            # print(f"Response before putting its json into queue : {response.json()}")
            q.put(response.json())
        return response.json()
    except (JSONDecodeError, AttributeError) as ex:
        if q is not None:
            q.put(response)
        return response


get_search_results.consequent_unknown_errors = 0


def df_to_csv(df, csv_filename):
    # Helper function, writes df to csv row by row
    # if os.path.exists(csv_filename):
    #     os.remove(csv_filename)
    try:
        with open(csv_filename, "w", encoding='utf-8', newline="") as f:
            writer = csv.writer(f)
            # print(f'Length of anime DB is : {len(df)}')
            print(df.index)
            writer.writerow(df.columns)
            for ind in df.index:
                print(ind)
                df.reset_index()
                print(df)
                print(df.loc[ind])
                writer.writerow(df.loc[ind])
    except PermissionError:
        # logger.error("Unable to convert df to CSV. Please close the file in Excel.")
        time.sleep(Sleep.LONG_SLEEP)
        df_to_csv(df, csv_filename)
        # Critical function, we must try again until it works since we're using the csv later.


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
        print("No such concatenation type")
        raise ValueError

    save_pickled_file(dict1_filename, dict1)


def remove_zero_columns(df : pl.DataFrame):
    zero_columns = [col for col in df.columns if df[col].sum() == 0]
    for col in zero_columns:
        if col.endswith("Affinity"):
            # If say, "Crime Affinity" is full of zeros, we want to remove it and the "Crime" column.
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
        # print("Warning, NaNs detected")
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
        # Base case: when data is neither a dict nor a list, just return the data itself
        return data


def is_redis_cache():
    # Access the backend of the default cache
    backend = settings.CACHES['default']['BACKEND']

    # Check if the backend is set to django_redis.cache.RedisCache
    return 'django_redis.cache.RedisCache' in backend


def redis_cache_wrapper(timeout):
    def decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            print("Entering cache decorator")
            print(*args, **kwargs)
            cache_key = f"{function.__name__}_{'_'.join(str(arg) for arg in args)}_{'_'.join(f'{key}_{value}' for key, value in kwargs.items())}"
            print("Cache key is", cache_key)
            result = cache.get(cache_key)
            if result is not None:
                return result
            print("Entering function")
            result = function(*args, **kwargs)

            if not (isinstance(result, dict) and 'error' in result):
                print("No error found, caching the result")
                cache.set(cache_key, result, timeout)
                print("5")
            return result
        return wrapped
    return decorator


def rate_limit(rate_lim=1, cache_key=None):
    def decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            print("Entering rate limiter")
            # nonlocal cache_key
            # if not cache_key:
            #     if "site" in kwargs.keys():
            #         cache_key = f"{kwargs['site']}_rate_limit"
            #     else:
            #         raise ValueError("If the decorator doesn't have a cache_key,"
            #                          "the wrapped function must provide a site")

            print("Cache key is", cache_key)
            while not can_make_api_call(rate_lim, cache_key):
                print("Sleeping to avoid rate limit")
                time.sleep(0.1)
            print("Entering function")
            result = function(*args, **kwargs)
            return result
        return wrapped
    return decorator


def can_make_api_call(rate_lim, cache_key):

    current_time = int(time.time())
    last_api_call_time = cache.get(cache_key)

    if last_api_call_time is None or (current_time - int(last_api_call_time)) >= rate_lim:
        # Update the last API call time and proceed
        cache.set(cache_key, current_time)
        return True
    else:
        # Wait if we're within the rate limit window
        return False


# def make_api_call_with_rate_limit(r: redis.Redis, key: str, rate_limit: int):
#     while not can_make_api_call(r, key, rate_limit):
#         time.sleep(0.1)  # Sleep a bit before trying again





# def cache_result_in_redis(function):
#     def inner(*args, **kwargs):
#         cache_key = f"{function.__name__}_{'_'.join(str(arg) for arg in args)}_{'_'.join(f'{key}_{value}' for key, value in kwargs.items())}"
#
#         result = cache.get(cache_key)
#         if result is not None:
#             return result
#
#         result = function(*args, **kwargs)
#
#         cache.set(cache_key, result, CACHE_TIMEOUT)
#         return result
#     return inner