import requests
import sys

# Client_details.txt must contain your MAL client ID on the first line,
# and your MAL client secret on the second line.
try:
    with open('Client_details.txt') as f:
        id_secret = f.readlines()
    CLIENT_ID = id_secret[0][:-1]
    CLIENT_SECRET = id_secret[1]
except IndexError:
    print("ERROR - Client_details.txt is not in the correct format")
    sys.exit(1)
except FileNotFoundError:
    print("ERROR - Client_details.txt not found")
    sys.exit(1)


def get_new_code_verifier() -> str:
    import secrets
    return secrets.token_urlsafe(100)[:128]


def get_initial_code_link(code_challenge):
    url = f"""https://myanimelist.net/v1/oauth2/authorize?response_type=code
    &client_id={CLIENT_ID}&code_challenge={code_challenge}"""
    print(f'Authorise your application by clicking here: {url}\n')


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
