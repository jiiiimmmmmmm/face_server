from flask import Flask, request, jsonify, render_template
import base64
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from utils import * 
app = Flask(__name__)

manager = face_manager(database_path="mysql://root:999123ac@127.0.0.1/student_exam")

@app.route('/search', methods=['POST','GET'])
def search():
    data = request.data
    data = json.loads(data)
    print(type(data['imageBase64']))
    image = base64.b64decode(data['imageBase64'])
    image = np.fromstring(image, np.uint8)
    image = image.reshape([data['height'],data['width'],data['channel']])
    image = image[:,:,:3]

    cv2.imwrite('temp.png', image)

    result = manager.face_nearest_face(image)
    print(result)


    res = {
            'msg': 'success',
            'id': '20172131096',
            'name': '吴梓祺',
            'exam': '计算机原理',
            'room': '主教学楼 西340',
            'time': '9:30-11:30'
        }
    return jsonify(res), 200 



@app.route('/welcome')
def welcome():
    return 'welcome'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')