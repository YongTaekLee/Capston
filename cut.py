import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# 이미지를 저장할 디렉토리
result_dir = r"C:\Users\YongTaek\Desktop\ex\ah11"

# image_dir은 14*1 사진의 경로및 파일이름.
def seperate(image_dir, result_dir=result_dir):
    # 이미지 불러오기
    img = cv2.imread(image_dir)

    # 가로로 긴 이미지를 잘라서 담을 리스트
    img_list = []

    # 리스트에 간격이 일정하게 자름.
    for i in range(14):
        img_list.append(img[0:256, 256*i:256*(i+1)])
    for i in range(14):
        #fin_image = cv2.resize(img_list[i],(512,512))
        fin_image = img_list[i]
        cv2.imwrite(os.path.join(result_dir,
        os.path.split(image_dir)[-1].split('.')[0] + '_'+ str(i) + '.jpg'), fin_image)
    return 0;
    
seperate(r'C:\Users\YongTaek\Desktop\ex\test11.jpg',result_dir=result_dir)
#img = cv2.imread(r'C:\Users\YongTaek\Desktop\caps_논문_gan\1\1-03.jpg')
#type(img)
