import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import lmdb
import struct
import base64
import numpy as np
import cv2
import glob
import tensorflow as tf
from keras.models import load_model
from annoy import AnnoyIndex


def loadKerasModel(file):
    model = load_model(file, compile=False)
    return model

def makeSquare(face):
    h,w,_ = face.shape
    if w>h:
        start = int((w-h)/2)
        end = start+h
        return face[:,start:end,:]
    else:
        start = int((h-w)/2)
        end = start+w
        return face[start:end,:,:]

def preprocess(face, required_size=(160, 160)):
    face = makeSquare(face)
    face = cv2.resize(face, required_size)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    # standardize pixel values across channels (global)
    ret = (np.float32(face) - 127.5) / 127.5
    ret = ret.reshape([-1,*required_size,3])
    return ret


def printAllFromLmdb():
    with lmdb.open('./lmdb') as env:
        wfp = env.begin()
        for key,value in wfp.cursor():
            print(decode(key,value))

def encode(key,vector):
    key = struct.pack('H',key)
    value = struct.pack('128f',*vector)
    return key,value
def decode(key, value):
    key = struct.unpack('H', key)
    vector = struct.unpack('128f',value)
    return key,vector
def writeEmbeddingsToLmdb(path):
    imagePaths = glob.glob(path)
    print(','.join([path.split('\\')[-1] for path in imagePaths]))
    embs = []
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = preprocess(image)
        emb = keras_model.predict(image)
        emb /= np.linalg.norm(emb)
        embs.append(emb[0])
    embs = np.stack(embs)
    with lmdb.open('./lmdb') as env:
        wfp = env.begin(write=True)
        for index,emb in enumerate(embs):
            key,value = encode(index, emb)
            wfp.put(key,value)
        wfp.commit()


def embedding(model, faceImage):
    sample = preprocess(faceImage)
    emb = model.predict(sample)[0]
    emb /= np.linalg.norm(emb)
    return emb

def createAnnoyIndex():
    annoy = AnnoyIndex(128, metric='angular')
    with lmdb.open('./lmdb') as env:
        wfp = env.begin()
        for key, value in wfp.cursor():
            key,vector = decode(key,value)
            annoy.add_item(key[0],vector)
    annoy.build(10)
    if not os.path.exists('./annoyIndex/'):
        os.mkdir('./annoyIndex/')
    annoy.save('./annoyIndex/face_vector.nn')
    print('create annoy index successfully')

if __name__ == '__main__':
    keras_file = "./model/facenet_keras.h5"
    keras_model = loadKerasModel(keras_file)
    # writeEmbeddingsToLmdb('./image_cropped/*.jpg')
    # printAllFromLmdb()
    # createAnnoyIndex()
    annoy = AnnoyIndex(128, metric='angular')
    annoy.load('./annoyIndex/face_vector.nn')
    face_image = cv2.imread('temp.png')
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    e = embedding(face_image)
    print(e)
    res = annoy.get_nns_by_vector(e,5,include_distances=True)
    print(res)

