import os
import time
import shutil

eyestate_index = 16
print('the current location: ' + os.getcwd())
root_folder = os.getcwd()
if('mrlEyes_2018_01' in os.listdir()):
    print('Success, folder found!')
    try:
        os.mkdir('closed')
        os.mkdir('open')
        print('directories')
    except:
        print('Failed directories')
else:
    print('failed, put this file in the same folder with the mrlEyes_2018_01 dataset')
    time.sleep(5)
    exit()
open_dir = root_folder + '\open'
closed_dir = root_folder + '\closed'
os.chdir(root_folder + '\mrlEyes_2018_01')
folders = os.listdir()
print(folders)

# transfare the data without dublication
for folder in folders:
    os.chdir(root_folder + '\mrlEyes_2018_01/' + folder)
    images = os.listdir()
    for image in images:
        if(image[eyestate_index] == '0'):
            shutil.move(image, closed_dir)
        else:
            shutil.move(image, open_dir)
else:
    print('Done!')