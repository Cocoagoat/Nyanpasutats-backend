import requests
import sys
from filenames import data_path
# Client_details.txt must contain your MAL client ID on the first line,
# and your MAL client secret on the second line.
try:
    with open(data_path / "Client_details.txt") as f:
        id_secret = f.readlines()
    CLIENT_ID = id_secret[0].strip()
    CLIENT_SECRET = id_secret[1].strip()
except IndexError:
    print("ERROR - Client_details.txt is not in the correct format")
    sys.exit(1)
except FileNotFoundError:
    print(5)
    print("ERROR - Client_details.txt not found")
    sys.exit(1)


def get_new_code_verifier() -> str:
    import secrets
    return secrets.token_urlsafe(100)[:128]


def get_initial_code_link(code_challenge):
    url = f"""https://myanimelist.net/v1/oauth2/authorize?response_type=code&client_id={CLIENT_ID}&code_challenge={code_challenge}"""
    print(f'Authorise your application by clicking here: {url}')


def get_auth_token(authentication_code, code_verifier):
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


def get_access_token():
    code_verifier = code_challenge = get_new_code_verifier()
    get_initial_code_link(code_verifier)
    authentication_code = input('Copy-paste the authorization code from the URL :').strip()
    access_token = get_auth_token(authentication_code, code_verifier)
    print(access_token)
    return access_token

# def get_headers():
#     headers = {}
#     # with open('Authorization.txt', "r") as f:
#     with open(auth_filename) as f:
#         headers_list = f.read().splitlines()
#         if headers_list:
#             headers['Authorization'] = headers_list[0]
#             headers['refresh_token'] = headers_list[1]
#         headers['content-type'] = 'application/x-www-form-urlencoded'
#     return headers