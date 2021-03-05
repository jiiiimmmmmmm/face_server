# face_server
 使用flask做人脸业务后端

# 安装环境

```cmd
pip install tensorflow==2.4
pip install annoy
pip install lmdb
pip install numpy
pip install matplotlib
pip install opencv-python
```

# 文件说明 

keras_model_test.ipynb 包括了解析base64图片文件，预处理图片，facenet向量化图片，向量保存到lmdb，用annoy添加索引，测试等工作

要运行代码，需要在根目录加入image文件夹和model文件夹