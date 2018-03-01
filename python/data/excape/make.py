import requests
import os
from hashlib import sha256

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

def download():
    urls = [
            ( 
                "1MI2gXqc0-PM77qxReYQrmO1QyAITMBZ0",
                "f49c480076ee43f5635bd957a07c44a8a06133d10985c683b9260930831eb163",
                "side_c2v.ddm",
            ),
            (
                "1UA1gQrnintr4BCWuIgg5LFda7Sx4fSAe",
                "8212af437e66ac4db2c48a7b0be63c8822ae803ceee7d080d03d8310d94b5849",
                "side_ecfp6_counts_var005.sdm",
            ),
            ( 
                "15QG_g9h7d5lqbnybOilb7UX08FFiHN9l",
                "7326d97c8546fb09eb4a4a6b54758046ff04e7402cf857484d8263006e119a56",
                "side_ecfp6_folded_dense.ddm",
            ),
            ( 
                "1OpvOLh0fwFQQRDyp8vYdAGU_CIy9WNCP",
                "9b8e458612a72d7051463d761248c54edfcb8bbfc73266536a5791fa5b047da2",
                "test.sdm",
            ),
            ( 
                "1TGq9qSkKa7fvnwdJ2g5drTTaNTwlnOYF",
                "f6d9315f2c905146275caa1e1b03e380d870c33166caec81c2bdb35a7efe77ef",
                "train.sdm",
            ),
    ]
 
    for id, expected_sha, output in urls:
        if os.path.isfile(output):
            actual_sha = sha256(open(output, "rb").read()).hexdigest()
            if (expected_sha == actual_sha):
                print("already have %s" % output)
                continue

        print("download %s" % output)
        download_file_from_google_drive(id, output)

if __name__ == "__main__":
    download()
