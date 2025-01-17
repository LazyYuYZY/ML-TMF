import json
import os


'''文件操作'''
class rw_files(object):
    @staticmethod
    def write_dict(txt_name, dicts):
        f = open(txt_name, 'w')
        json_dicts = json.dumps(dicts, indent=1)
        f.write(txt_name + '\n' + json_dicts)
        f.close()

    @staticmethod
    def get_dict(json_file_name):
        # 读取JSON文件并解析为字典对象
        with open(json_file_name, 'r') as json_file:
            lines = json_file.readlines()[1:]  # 读取所有行并从第二行开始
            json_data = json.loads("".join(lines))
        return json_data

    @staticmethod
    def change_name(folder_path, old_part, new_part, del_l=False, del_r=False):
        files = os.listdir(folder_path)  # 获取文件夹下的所有文件名

        for file in files:
            file_path = os.path.join(folder_path, file)  # 获取文件的完整路径
            tlist = file.split(old_part)
            if len(tlist) <= 1:
                return
            if del_l:
                new_file_name = new_part + tlist[1]
            if del_r:
                new_file_name = tlist[0] + new_part
            # new_file_name=file_path+new_part
            new_file_path = os.path.join(folder_path, new_file_name)  # 构建新文件的完整路径
            # 移动文件并重命名
            os.rename(file_path, new_file_path)