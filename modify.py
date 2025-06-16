import os
import shutil

def rename_files_and_update_txt(data_dir='', txt_dir=''):
   # 遍历train/val/test目录
   for split in ['train', 'val', 'test']:
       # 重命名文件夹
       split_dir = os.path.join(data_dir, split)
       for folder in os.listdir(split_dir):
           old_path = os.path.join(split_dir, folder)
           new_folder = folder.replace(' ', '_')
           new_path = os.path.join(split_dir, new_folder)
           if ' ' in folder:
               shutil.move(old_path, new_path)
       
       # 更新txt文件
       txt_path = os.path.join(txt_dir, f'{split}.txt')
       if os.path.exists(txt_path):
           with open(txt_path, 'r') as f:
               lines = f.readlines()
           
           with open(txt_path, 'w') as f:
               for line in lines:
                   path, label = line.rsplit(' ', 1)
                   new_path = path.replace(' ', '_')
                   f.write(f'{new_path} {label}')

if __name__ == '__main__':
   rename_files_and_update_txt(
       data_dir='./MDCS/data/RSD46-WHU_LT',
       txt_dir='./MDCS/data_txt/RSD46-WHU_LT'
   )