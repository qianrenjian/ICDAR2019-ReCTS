#txt_path2 = 'H:/why_workspace/ReCTS/Task1/result_resnet_enhance.txt'
txt_path1 = 'H:/why_workspace/ReCTS/Task1/result_resnet_enhance_count5.txt'
txt_path2 = 'H:/why_workspace/ReCTS/Task1/result_resnet_enhance_count5.txt'

f1 = open(txt_path1, 'r', encoding='utf-8')
f2 = open(txt_path2, 'r', encoding='utf-8')

line1 = f1.readlines()
line2 = f2.readlines()
# line1 = f1.readline()
# line2 = f2.readline()
#line1 = line1.strip('\n')
print(line1[0])
print(line1[1])
print(line1[0][0])
print(line1[1][0])
# for i in range(0, 29335):
#     if line1[i]!=line2[i]:
#         print(line1[i])
#         print(line2[i])

