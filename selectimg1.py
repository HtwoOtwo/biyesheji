import shutil
import os
import random
#new img path
file_path = '/data'

#current_foder是‘模拟’文件夹下所有子文件名组成的一个列表
current_path='USAFtarget/50'
current_folder = os.listdir(current_path)#current_foder是‘模拟’文件夹下所有子文件名组成的一个列表
def show_files(path, all_files):
    file_list = os.listdir(path)
    for file in file_list:
        cur_path = os.path.join(path, file)
        if os.path.isdir(cur_path):
            show_files(cur_path, all_files)
        else:
            all_files.append(cur_path)
    return all_files
current_folder = show_files(current_path, [])
sort(current_folder)
#random selenct 10000 imgs from 50000 imgs
#random.shuffle(current_folder)
holo = []
origin=[]
cnt=0
for e in current_folder:
    e.split('/')[-1][0]=='h':
        if cnt%30==0:
            holo.append(e)
            org = e.replace('holo','original')
            origin.append(org)
        cnt+=1



# 第二部分，将名称为file的文件复制到名为file_dir的文件夹中
for i,holo, org in zip(range(1000),holo,origin):
    #拼接出要存放的文件夹的路径
    target_path = file_path + '/holo/' + str(i)+'.bmp'
    print(target_path)
    #将指定的文件file复制到file_dir的文件夹里面
    shutil.copy(holo,target_path)
    target_path = file_path + '/original/' + str(i)+'.bmp'
    # 将指定的文件file复制到file_dir的文件夹里面
    shutil.copy(org,target_path)
