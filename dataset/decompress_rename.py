import os
import zstandard as zstd

def decompress_zst_files(folder_path):
    # 获取指定文件夹内所有的.zst文件
    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    files.sort()  # 对文件名进行排序，确保按文件名顺序处理

    # 遍历文件，解压并重命名
    for idx, filename in enumerate(files, start=1):
        zst_path = os.path.join(folder_path, filename)
        output_path = os.path.join(folder_path, f"{idx}.txt")

        # 打开.zst文件
        with open(zst_path, 'rb') as compressed:
            dctx = zstd.ZstdDecompressor()
            # 创建一个解压缩的上下文并读取压缩文件
            with dctx.stream_reader(compressed) as reader:
                # 读取所有解压后的数据并写入新文件
                with open(output_path, 'wb') as decompressed:
                    decompressed.write(reader.read())

        print(f"Decompressed and renamed '{zst_path}' to '{output_path}'")

# 使用示例
folder_path = 'original_data/'
decompress_zst_files(folder_path)

