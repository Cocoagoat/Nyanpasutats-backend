# This is a sample Python script.
import requests
import json
import time
from general_utils import terminate_program
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


try:
    with open('Client_details.txt') as f:
        id_secret = f.readlines()
    CLIENT_ID = id_secret[0]
    CLIENT_SECRET = id_secret[1]
except IndexError:
    print("ERROR - Client_details.txt is not in the correct format")
    terminate_program()
except ValueError:
    print("ERROR - Client_details.txt not found. Please get a client ID and client secret"
          "and write them down in a .txt file in two lines.")




def get_new_code_verifier() -> str:
    import secrets
    return secrets.token_urlsafe(100)[:128]


def get_initial_code_link(code_challenge):
    url = f"""https://myanimelist.net/v1/oauth2/authorize?response_type=code&client_id={CLIENT_ID}&code_challenge={code_challenge}"""
    print(f'Authorise your application by clicking here: {url}\n')


def get_auth_token(authentication_code, code_verifier):
    redirect_url = "https://myanimelist.net/profile/BaronBrixius"
    url = 'https://myanimelist.net/v1/oauth2/token'
    data = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'code': authentication_code,
        'code_verifier': code_verifier,
        'grant_type': 'authorization_code'
    }
    response = requests.post(url, data).json()
    print(response)
    return response
    # return response['access_token']


def get_access_token():
    code_verifier = code_challenge = get_new_code_verifier()
    get_initial_code_link(code_verifier)
    authentication_code = input('Copy-paste the authorization code from the URL :').strip()
    access_token = get_auth_token(authentication_code, code_verifier)
    print(access_token)
    return access_token


# token = get_auth_token()
# print(token)
# code_verifier = code_challenge = get_new_code_verifier()
# get_initial_code_link(code_verifier)
# authentication_code = input('Copy-paste the authorization code from the URL :').strip()
# access_token = get_auth_token(authentication_code, code_verifier)
# print(access_token)
