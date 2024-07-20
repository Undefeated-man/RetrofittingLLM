import os

def merge_txt_files_in_groups(folder_path, group_size=3):
    # 获取指定文件夹内所有的.txt文件
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    txt_files.sort()  # 对文件名进行排序，确保按文件名顺序处理

    # 创建合并后的文件
    group_number = 1
    for i in range(1, len(txt_files), group_size):
        grouped_files = txt_files[i:i+group_size]  # 获取一组文件名
        output_file_path = os.path.join(folder_path, f'train_{group_number}.txt')
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            # 依次读取每个文件并写入到输出文件
            for filename in grouped_files:
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    output_file.write(file.read() + '\n')  # 将文件内容写入输出文件
        print(f'Created "{output_file_path}" with {len(grouped_files)} files.')
        group_number += 1

# 使用示例
folder_path = 'processed'
merge_txt_files_in_groups(folder_path)

