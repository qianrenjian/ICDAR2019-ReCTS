import os
path = 'H:/why_workspace/ReCTS/img_dir2/'
sub_file = os.listdir(path)
img = []
count = 0
for root, dirs, files in os.walk(path):
    #print(files)
    for file in files:
        #print(file)
        count+=1
print(count)

#print(len(img))