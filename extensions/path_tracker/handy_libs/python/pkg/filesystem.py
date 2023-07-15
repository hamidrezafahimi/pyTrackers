import os

def get_file_list(dir, ext):
    file_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(ext):
                file_list.append(os.path.join(root, file))
    return file_list
