import os
name_list = []
with open("image_list.txt") as f:
    for line in f:
        name_list.append(line.strip())
for name in os.listdir("img"):
    if name not in name_list:
        os.remove(os.path.join("img", name))
label_list = []
with open("label_list.txt") as f:
    for line in f:
        label_list.append(line.strip())
for name in os.listdir("labels"):
    if name not in label_list:
        os.remove(os.path.join("labels", name))
print(label_list)
