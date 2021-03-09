from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/search', methods=['POST','GET'])
def search():
    data = request.data
    data = json.loads(data)
    image = base64.b64decode(data['image'])
    image = np.fromstring(image, np.uint8)
    image = image.reshape([data['height'],data['width'],data['channel']])
    image = image[:,:,:3]

    # show image
    # imageShow = image / 255
    # plt.imshow(imageShow)
    # plt.show()
    
    # to do
    # preprocess
    # predict
    # find the closest face
    # query database to get student information

    res = {
        'msg': 'success',
        'id': '20172131096',
        'name': '吴梓祺',
        'exam': '计算机原理',
        'room': '主教学楼 西340',
        'time': '9:30-11:30'
    }
    return jsonify(res)

@app.route('/welcome')
def welcome():
    return 'welcome'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')