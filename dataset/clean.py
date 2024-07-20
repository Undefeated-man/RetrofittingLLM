import os

for file in os.listdir("."):
    if os.path.exists(file) and not file.endswith(".py") and not file.endswith(".sh") and not file.endswith(".out") and not os.path.isdir(file):
        os.remove(file)
        print(f"File {file} has been deleted.")

