'''for i in file_list:
    image_dir = os.path.join(img_dir,i)
    image = Image.open(image_dir)
    img = image.crop((80,0,560,480))
    img.save(os.path.join(result_dir,i))
'''
import os
from PIL import Image
# 이미지를 저장할 디렉토리
result_dir = r"C:\Users\YongTaek\Desktop"

# image_dir은 14*1 사진의 경로및 파일이름.
def seperate(image_dir, result_dir=result_dir):
    # 이미지 불러오기
    img = Image.open(image_dir)

    # 가로로 긴 이미지를 잘라서 담을 리스트
    img_list = []

    # 리스트에 간격이 일정하게 자름.
    for i in range(14):
        img_list.append(img.crop((256*i,0,256*(i+1),256)))
    for i in range(14):
        #fin_image = cv2.resize(img_list[i],(512,512))
        fin_image = img_list[i]
        fin_image.save(os.path.join(result_dir,
        os.path.split(image_dir)[-1].split('.')[0] + '_'+ str(i) + '.jpg'))
    return 0;

seperate(r'C:\Users\YongTaek\Desktop\test8.jpg',result_dir=result_dir)

    