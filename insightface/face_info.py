import cv2
import os
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from numpy.linalg import norm

class FaceInfo:
    def __init__(self):
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640)) #ctx_id cpu -1 gpu 1

    #输出性别，年龄，脸部坐标
    def get_face_info(self, image_file, save_face_image=False, face_draw=False):
        face_info=[]
        #判断文件是否存在
        if os.path.exists(image_file):
            pass
        else:
            # print('%s file not exist'%image_file)
            return "%s file not exist"%image_file
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
        if len(faces) <=0:
            return "no face"
        #TODO:如果整张图是一张大脸，是找不到脸的
        #脸部画框保存下来
        if face_draw:
            rimg = self.app.draw_on(img, faces)
            rect_img = os.path.join(image_file_dir, image_file_name +"_rect.jpg")
            cv2.imwrite(rect_img, rimg)
        #把脸的信息写入数组
        for i in range(len(faces)):
            face = faces[i]
            # print(face)
            box = face.bbox.astype(int)
            if save_face_image:
                save_face = os.path.join(image_file_dir, image_file_name +"_"+ str(i) + ".jpg")
                # print(save_face)
                face_image = img[box[1]:box[1]+(box[3]-box[1]),box[0]:box[0]+(box[2]-box[0])]
                cv2.imwrite(save_face,face_image)
            # print(face.sex,face.age, [box[0], box[1], box[2], box[3])
            face_info.append({"sex":face.sex,"age":face.age, "rect":[box[0], box[1], box[2], box[3]]})
        #返回结果
        return face_info
    
    def compare_face(self, image_file1, image_file2, threshold=0.6):
        #判断文件是否存在
        if os.path.exists(image_file1) and os.path.exists(image_file2):
            pass
        else:
            # print('%s file not exist'%image_file)
            return "%s,%s file not exist"%(image_file1, image_file2)
        #TODO:人脸对齐
        #分析人脸
        img1 = cv2.imread(image_file1)
        faces1 = self.app.get(img1)
        if len(faces1) <=0:
            return "%s have no face"%image_file1
        img2 = cv2.imread(image_file2)
        faces2 = self.app.get(img2)
        if len(faces2) <=0:
            return "%s have no face"%image_file2
        #计算相似度
        faces1_embedding = faces1[0].normed_embedding
        faces2_embedding = faces2[0].normed_embedding
        score = np.dot(faces1_embedding, faces2_embedding)/(norm(faces1_embedding)*norm(faces2_embedding))
        print(score)
        if score > threshold:
            return "yes"
        else:
            return "no"        

if __name__ == "__main__":
    face = FaceInfo()
    # result=face.get_face_info("./t1.jpg", save_face_image=True)
    # print(result)
    # result=face.get_face_info("./t2.jpeg", save_face_image=True)
    # print(result)
    # result=face.get_face_info("./t3.jpeg", save_face_image=True)
    # print(result)
    # result=face.get_face_info("./t4.jpg", save_face_image=True, face_draw=True)
    # print(result)
    result=face.compare_face("./t4.jpg", "./t4_rect.jpg")
    print(result)
