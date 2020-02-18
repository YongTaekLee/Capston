import matplotlib.pyplot as plt
import STARGAN_BOTH
from STARGAN_BOTH import test_multi

image_dir = r'C:\Users\YongTaek\Desktop\test\test11.jpg'

result_dir = r'C:\Users\YongTaek\Desktop'

test_multi(image_dir,result_dir=result_dir, c_org1=[1,0,0,0,1],c_org2=[0,0,0,0,0,1,0,0])
    # c_org1 = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
    # c_org2 = angry, contemptuous, disgusted, fearful, happy, neutral, sad, surprised      