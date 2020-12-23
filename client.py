import requests
import sys

# initialize the Keras REST API endpoint URL along with the input
# image path
REST_API_URL = "http://localhost:8080/predict"
#IMAGE_PATH = r"images\dog.jpg"

# load the input image and construct the payload for the request
#image = open(IMAGE_PATH, "rb").read()
image = open(sys.argv[1], "rb").read()
payload = {"image": image}

# submit the request
r = requests.post(REST_API_URL, files=payload).json()

# ensure the request was successful
if r["success"]:
    # loop over the predictions and display them
    for (i, result) in enumerate(r["predictions"]):
        print("{}. {}: {:.4f}".format(i + 1, result["label"],
            result["probability"]))

# otherwise, the request failed
else:
    print("Request failed")