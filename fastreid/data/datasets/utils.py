import os


def find_jpg_files(directory):
    jpg_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg"):
                jpg_files.append(os.path.join(root, file))
    sorted_jpg_files = sorted(jpg_files)
    return sorted_jpg_files


def read_txt_and_convert_to_list(file_path):
    # 打开文件
    with open(file_path, 'r') as file:
        content = file.read()
    # 去除首尾的空白字符，确保字符串干净
    cleaned_content = content.strip()
    # 使用逗号分割文本内容，并转换为整数列表（根据需求调整为浮点或其他类型）
    number_list = list(map(int, cleaned_content.split(',')))
    return number_list