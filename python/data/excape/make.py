import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

links = {
    "side_c2v.ddm": "1MI2gXqc0-PM77qxReYQrmO1QyAITMBZ0",
    "side_ecfp6_counts_var005.sdm": "1UA1gQrnintr4BCWuIgg5LFda7Sx4fSAe",
    "side_ecfp6_folded_dense.ddm": "15QG_g9h7d5lqbnybOilb7UX08FFiHN9l",
    "test.sdm": "1OpvOLh0fwFQQRDyp8vYdAGU_CIy9WNCP",
    "train.sdm": "1TGq9qSkKa7fvnwdJ2g5drTTaNTwlnOYF",
}

for dest, id in links.items():
    download_file_from_google_drive(id, dest)

