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
from mtcnn import MTCNN
import matplotlib.pyplot as plt



class faceHelper:
    @staticmethod
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

    @classmethod
    def preprocess_single_RGB_image(cls, face, required_size=(160,160)):
        assert face.shape[-1] == 3
        face = cls.makeSquare(face)
        face = cv2.resize(face, required_size)
        face = (np.float32(face) - 127.5) / 127.5
        face = np.expand_dims(face, 0)
        return face

    @staticmethod
    def encode_embedding(vector, length = 128):
        value = struct.pack('{}f'.format(length),*vector)
        return value
    @staticmethod
    def decode_embedding(value, length = 128):
        vector = struct.unpack('{}f'.format(length), value)
        return vector



class faceEmbedding:
    def __init__(self, model_path):
        """初始化 载入模型"""
        self.model = load_model(model_path, compile=False)
    
    def get_face_embedding(self, face):
        face = faceHelper.preprocess_single_RGB_image(face)
        embedding =  self.model.predict(face)
        embedding /= np.linalg.norm(embedding)
        return embedding[0]


from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, DateTime, String, ForeignKey, BINARY, desc
from sqlalchemy.orm import sessionmaker

Base  = declarative_base()
class StudentFace(Base):
        __tablename__ = 'student_face'

        id = Column(Integer, primary_key=True)
        _face_embedding = Column('face_embedding',BINARY(12), nullable=False)
        student_id = Column(Integer , nullable=False, index=True)  

        def __init__(self, id, face_embedding, student_id):
            self.id = id
            self._face_embedding = faceHelper.encode_embedding(face_embedding)
            self.student_id = student_id

        @property
        def face_embedding(self):
            return faceHelper.decode_embedding(self._face_embedding)

class faceDetection:
    def __init__(self):
        self.model = MTCNN()
    
    def detect_face_from_RGB(self, face_image):
        result = self.model.detect_faces(face_image)
        if len(result) == 1 and result[0]['confidence'] > 0.99:
            x,y,w,h = map(lambda value: abs(value), result[0]['box'])
            return face_image[y:y+h, x:x+w]
        else:
            return np.array([])
    
class face_manager:

    def __init__(self, database_path, face_net_model_path = 'model/facenet_keras.h5', annoy_index_dir='annoyIndex'):
        engine = create_engine(database_path)
        Session = sessionmaker(bind=engine)
        self.session = Session()
        self.face_embedding = faceEmbedding(face_net_model_path)
        self.face_detector = faceDetection()
        self.last_id = self.session.query(StudentFace.id).order_by(desc(StudentFace.id)).first()
        self.last_id = self.last_id[0] if self.last_id else 0
        self.annoy = None
        self.annoy_index_dir = annoy_index_dir


    def add_face(self, face_image, student_id):
        face_image = self.face_detector.detect_face_from_RGB(face_image)
        if face_image.shape == (0,):
            print('do not detect any face')
            return face_image
        embedding = self.face_embedding.get_face_embedding(face_image)
        stuFace = StudentFace(self.last_id+1 , embedding, student_id)
        try:
            self.session.add(stuFace)
            self.session.commit()
            self.last_id += 1
            return face_image
        except Exception as e:
            print(e.__traceback__)
            print(e)
            self.session.rollback()

    def create_annoy_index(self, vector_length=128, annoy_index_dir='annoyIndex', tree_num=50):
        self.vector_length = vector_length
        self.annoy_index_dir = annoy_index_dir
        self.tree_num = tree_num

        annoy = AnnoyIndex(vector_length, metric='angular')
        for student_face in self.session.query(StudentFace).all():
            annoy.add_item(student_face.id, student_face.face_embedding)

        annoy.build(tree_num)
        self.annoy = annoy

        if not os.path.exists(annoy_index_dir):
            os.mkdir(annoy_index_dir)
        annoy.save(os.path.join(annoy_index_dir,'face_vector.nn'))
        print('create annoy index successfully')

    def face_nearest_face(self, face_image, n_nearest=5, show_distance=True):
        if not self.annoy:
            self.annoy = AnnoyIndex(128, metric='angular')
            self.annoy.load(os.path.join(self.annoy_index_dir,'face_vector.nn'))

        face_image = self.face_detector.detect_face_from_RGB(face_image)
        if face_image.shape == (0,) :
            print('do not detect any face')
            return None

        embedding = self.face_embedding.get_face_embedding(face_image)
        ids, distance = self.annoy.get_nns_by_vector(embedding, n_nearest ,include_distances=show_distance)
        ids = [self.session.query(StudentFace.student_id).filter_by(id=id).first()[0] for id in ids]
        return (ids, distance)



if __name__ == '__main__':

    manager = face_manager(database_path="mysql://root:999123ac@127.0.0.1/student_exam")

    student_image_paths = glob.glob('data/image/selected/*')

    for student_image_path in student_image_paths:
        face_image = cv2.imread(student_image_path)
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        id = int(os.path.basename(student_image_path).split('_')[0])
        face_image = manager.add_face(face_image, student_id=id)
        
        if face_image.shape != (0,):
            print(id)
            face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            target = os.path.join('data/image/casia_facev5_cropped/',os.path.basename(student_image_path))
            print(target)
            cv2.imwrite(target, face_image)
     
    manager.create_annoy_index()

    # image_paths = glob.glob('data/image/casia_facev5/*/*')
    # image_paths = image_paths[:20*5]
    # for i, image_path in enumerate(image_paths):
    #     print(os.path.basename(image_path))
    #     face_image = cv2.imread(image_path)
    #     face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    #     result = manager.face_nearest_face(face_image)
        
    #     print(result)
