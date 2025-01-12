import os

def scan_directory_for_file_types(directory_path):
    # 存储不同类型文件的计数
    file_types = {}
    # 存储每种类型文件的路径
    file_paths = {}

    # 遍历目录及子目录
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # 获取文件扩展名 (小写)
            file_extension = file.split('.')[-1].lower() if '.' in file else 'no_extension'
            file_path = os.path.join(root, file)

            # 更新文件类型计数
            if file_extension in file_types:
                file_types[file_extension] += 1
                file_paths[file_extension].append(file_path)
            else:
                file_types[file_extension] = 1
                file_paths[file_extension] = [file_path]

    # 输出每种类型文件的数量及路径
    print("文件类型统计：")
    for ext, count in file_types.items():
        print(f"{ext.upper() if ext != 'no_extension' else '无扩展名'} 文件: {count} 个")
        # for path in file_paths[ext]:
        #     print(f"  - {path}")

if __name__ == "__main__":
    # 替换为你要扫描的目录路径
    directory_path = "C:/Users/Dingz/Desktop/273"
    scan_directory_for_file_types(directory_path)
