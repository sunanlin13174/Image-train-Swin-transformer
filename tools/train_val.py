import os
import shutil
data_path = 'E:/meprint_data23'
save_path = 'E:/data_copy'

data_list = os.listdir(data_path)
data_list.sort()
print(data_list)
for index,folder in enumerate(data_list):
    folder_path = os.path.join(data_path,folder)

    os.mkdir(save_path+'/{}'.format(index))
    for i,img in enumerate(os.listdir(folder_path)):
        img_path = os.path.join(data_path,folder_path,img)
        if i<800:
           shutil.copy(img_path,save_path+'/{}'.format(index))
