from PIL import Image
import os
import matplotlib.pyplot as plt


img_dir = r'C:\Users\YongTaek\Desktop\originalimages_part3'
result_dir = r'C:\Users\YongTaek\Desktop\set'
file_list = os.listdir(img_dir)
for i in file_list:
    image_dir = os.path.join(img_dir,i)
    image = Image.open(image_dir)
    img = image.crop((80,0,560,480))
    img.save(os.path.join(result_dir,i))


img_dir = r'C:\Users\YongTaek\Desktop\caps_논문\set'
file_list = os.listdir(img_dir)
for i in file_list:
    if '-01' in i.split('.')[0] :
        os.remove(os.path.join(img_dir,i))
    elif '-02' in i.split('.')[0] :
        os.remove(os.path.join(img_dir,i))
    elif '-09' in i.split('.')[0] :
        os.remove(os.path.join(img_dir,i))
    elif '-10' in i.split('.')[0] :
        os.remove(os.path.join(img_dir,i))
    elif '-14' in i.split('.')[0] :
        os.remove(os.path.join(img_dir,i))