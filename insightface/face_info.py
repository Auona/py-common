import cv2
import os
import numpy as np
import insightface
from insightface.app import FaceAnalysis

class FaceInfo:
    def __init__(self):
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    #输出性别，年龄，脸部坐标
    def get_face_info(self, image_file, save_face_image=False):
        face_info=[]
        #判断文件是否存在
        if os.path.exists(image_file):
            pass
        else:
            print('%s file not exist'%image_file)
            return face_info
        #获取图片路径
        image_file_dir = os.path.dirname(image_file)
        # print(image_file_dir)
        #获取图片名字、后缀
        image_file_name = os.path.basename(image_file)
        image_file_name = os.path.splitext(image_file_name)[0]
        #读取图片
        img = cv2.imread(image_file)
        #获取脸信息
        faces = self.app.get(img)
        #把脸的信息写入数组
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(int)
            if save_face_image:
                save_face = os.path.join(image_file_dir, image_file_name +"_"+ str(i) + ".jpg")
                print(save_face)
                face_image = img[box[1]:box[1]+(box[3]-box[1]),box[0]:box[0]+(box[2]-box[0])]
                cv2.imwrite(save_face,face_image)
            # print(face.sex,face.age, [box[0], box[1], box[2], box[3])
            face_info.append({"sex":face.sex,"age":face.age, "rect":[box[0], box[1], box[2], box[3]]})
        #返回结果
        return face_info

if __name__ == "__main__":
    face = FaceInfo()
    face_info=face.get_face_info("./t1.jpg", save_face_image=True)
    print(face_info)
