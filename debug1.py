import os

def list_files_in_directory(directory):
    try:
        # 使用 os.listdir 获取目录中的文件列表
        files = os.listdir(directory)

        # 打印文件列表
        print(f"Files in {directory}:")
        for file in files:
            print(file)
        return files
    except Exception as e:
        print(f"An error occurred: {e}")

# 用法示例
folder_path = 'data/'
filenames = list_files_in_directory(folder_path)
print(filenames)