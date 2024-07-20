import subprocess

command = ["nvidia-smi"]

result = subprocess.run(command, capture_output=True, text=True)

with open("out.out", "a") as f:
    f.write(result.stdout)

print("标准输出:")
print(result.stdout)

if result.stderr:
    print("标准错误:")
    print(result.stderr)

