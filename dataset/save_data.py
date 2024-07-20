import os
import time
import random
import threading
import subprocess

max_concurrent_downloads = 5
semaphore = threading.Semaphore(max_concurrent_downloads)

def download_file(url, filename):
    global cnt
    with semaphore:
        # 使用wget下载文件
        command = ['wget', '-O', filename, url]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print(f"Error downloading {filename}: {result.stderr}")
        cnt += 1
    else:
        print(f"Downloaded {filename} successfully.")


threads = []
cnt = 0
ref = {"train": "train", "validation": "holdout", "test": "holdout"}

for chunk in range(1, 11):
    for split in ["train", "validation", "test"]:
        for i in range(5000):
            try:
                url = f"https://huggingface.co/datasets/cerebras/SlimPajama-627B/resolve/main/{split}/chunk{chunk}/example_{ref[split]}_{i}.jsonl.zst?download=true"
                thread = threading.Thread(target=download_file, args=(url, f"{chunk}_{split}_{i}.json"))
                threads.append(thread)
                thread.start()
                time.sleep(random.uniform(0.5, 1))
                if cnt > 0:
                    os.remove(f"{chunk}_{split}_{i}.json")
                if cnt > 5:
                    cnt = 0
                    break
                # os.system(f"wget -O {chunk}_{split}_{i} https://huggingface.co/datasets/cerebras/SlimPajama-627B/resolve/main/{split}/chunk{chunk}/example_{split}_{i}.jsonl.zst?download=true")
            except Exception as e:
                print(f"Failed to download {chunk}_{split}_{i}: {e}")
                continue

for thread in threads:
    thread.join()
