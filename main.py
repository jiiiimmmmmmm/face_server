from flask import Flask, request, jsonify, render_template
import base64
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from utils import * 
app = Flask(__name__)

keras_file = "./model/facenet_keras.h5"
keras_model = loadKerasModel(keras_file)
annoy = AnnoyIndex(128, metric='angular')
annoy.load('./annoyIndex/face_vector.nn')

@app.route('/search', methods=['POST','GET'])
def search():
    data = request.data
    data = json.loads(data)
    image = base64.b64decode(data['imageBase64'])
    image = np.fromstring(image, np.uint8)
    image = image.reshape([data['height'],data['width'],data['channel']])
    image = image[:,:,:3]


    imageBGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img2gray = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2GRAY)
    imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()
    print('清晰度：'+str(imageVar))
    cv2.imwrite('temp.png',imageBGR)


    e = embedding(keras_model, image)
    print(annoy.get_nns_by_vector(e,5,include_distances=True))
    [id,dis] = annoy.get_nns_by_vector(e,1,include_distances=True)

    if dis[0] < 0.6:
        if id[0] < 4:
            res = {
                'msg': 'success',
                'id': '20172131096',
                'name': '吴梓祺',
                'exam': '计算机原理',
                'room': '主教学楼 西340',
                'time': '9:30-11:30'
            }
        else:
            res = {
                'msg': 'success',
                'id': '20172131053',
                'name': '朱晓健',
                'exam': '网络工程',
                'room': '南教学楼 西222',
                'time': '10:00-12:30'
            }
        return jsonify(res), 200 
    else:
        res = {
            'msg': 'unkown',
        }
        return jsonify(res), 201


@app.route('/welcome')
def welcome():
    return 'welcome'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')