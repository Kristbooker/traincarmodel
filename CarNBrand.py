import base64
import pickle
import cv2
import numpy
import requests  # Correct import for making HTTP requests
import os

def img2vec(img):
    resized = cv2.resize(img,(128,128), cv2.INTER_AREA)
    v, buffer = cv2.imencode(".jpg", resized)
    img_str = base64.b64encode(buffer).decode('utf-8')
    img_data_string = "data:image/jpeg;base64," + img_str
    
    # url = "http://127.0.0.1:8000/api/genhog"
    # or
    url = "http://localhost:8080/api/genhog"

    
    
    response = requests.get(url, json={"img_data":img_data_string})
    
    if response.status_code == 200:
        try:
            return response.json()
        except requests.exceptions.JSONDecodeError as e:
            print("JSON Decode Error:", e)
    else:
        print("API Request Error. Status Code:", response.status_code)
        return None

data_path="Cars Dataset/train"
# data_path="Cars Dataset/test"
X=[]
Y=[]

for sub in os.listdir(data_path):
    sub_path = os.path.join(data_path, sub)
    if os.path.isdir(sub_path):
        for fn in os.listdir(sub_path):   
            if not fn.startswith('.'):
                img_file_name = os.path.join(sub_path, fn)
                img = cv2.imread(img_file_name)
                X.append(img)
                Y.append(sub)

HOGVectors=[]

for i in range(len(X)):
    try:
        res = img2vec(X[i])
        if res is not None and "HOG VECTOR " in res:
            vec = res["HOG VECTOR "]
            vec.append(Y[i])
            HOGVectors.append(vec)
        else:
            print("API response format error or missing 'HOG VECTOR ' key:", res)
    except Exception as e:
        print("Error processing image:", e)

write_path_train="hogvectors_train.pkl"
# write_path_test="hogvectors_test.pkl"

pickle.dump(HOGVectors, open(write_path_train,"wb"))
# pickle.dump(HOGVectors, open(write_path_test,"wb"))
print("done")