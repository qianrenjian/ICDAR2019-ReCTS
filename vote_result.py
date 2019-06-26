import os

txt_path7 = 'H:/why_workspace/ReCTS/Task1/result_alexnet.txt'
txt_path6 = 'H:/why_workspace/ReCTS/Task1/result_alexnet_enhance.txt'
txt_path5 = 'H:/why_workspace/ReCTS/Task1/result_resnet.txt'
txt_path4 = 'H:/why_workspace/ReCTS/Task1/result_alexnet_enhance_decay.txt'
txt_path3 = 'H:/why_workspace/ReCTS/Task1/result_alexnet_enhance_count5.txt'
txt_path2 = 'H:/why_workspace/ReCTS/Task1/result_resnet_enhance.txt'
txt_path1 = 'H:/why_workspace/ReCTS/Task1/result_resnet_enhance_count5.txt'

f2 = open(txt_path2, 'r', encoding='utf-8')
f1 = open(txt_path1, 'r', encoding='utf-8')
f3 = open(txt_path3, 'r', encoding='utf-8')
f4 = open(txt_path4, 'r', encoding='utf-8')
f5 = open(txt_path5, 'r', encoding='utf-8')
f6 = open(txt_path6, 'r', encoding='utf-8')
f7 = open(txt_path7, 'r', encoding='utf-8')

result2 = f2.readlines()
result1 = f1.readlines()
result3 = f3.readlines()
result4 = f4.readlines()
result5 = f5.readlines()
result6 = f6.readlines()
result7 = f7.readlines()
count = 0
vote1 = 0
vote2 = 0
vote3 = 0
vote4 = 0
vote5 = 0
vote6 = 0
vote7 = 0
def cmp(num1, num2):
    if num1==num2:
        return True
    else:
        return False
txt_path = 'H:/why_workspace/ReCTS/Task1/result.txt'
with open(txt_path, "a", encoding='utf-8') as f:
    for i in range(0, 29335):
        cmp1 = cmp(result1[i], result2[i])
        cmp2 = cmp(result1[i], result3[i])
        cmp3 = cmp(result1[i], result4[i])
        cmp4 = cmp(result1[i], result5[i])
        cmp5 = cmp(result1[i], result6[i])
        cmp6 = cmp(result1[i], result7[i])
        cmp7 = cmp(result2[i], result3[i])
        cmp8 = cmp(result2[i], result4[i])
        cmp9 = cmp(result2[i], result5[i])
        cmp10 = cmp(result2[i], result6[i])
        cmp11 = cmp(result2[i], result7[i])
        cmp12 = cmp(result3[i], result4[i])
        cmp13 = cmp(result3[i], result5[i])
        cmp14 = cmp(result3[i], result6[i])
        cmp15 = cmp(result3[i], result7[i])
        cmp16 = cmp(result4[i], result5[i])
        cmp17 = cmp(result4[i], result6[i])
        cmp18 = cmp(result4[i], result7[i])
        cmp19 = cmp(result5[i], result6[i])
        cmp20 = cmp(result5[i], result7[i])
        cmp21 = cmp(result6[i], result7[i])
        if (cmp1==False)|(cmp2==False)|(cmp3==False)|(cmp4==False)|(cmp5==False)|(cmp6==False)|(cmp7==False)|(cmp8==False)|(cmp9==False)|(cmp10==False)|(cmp11==False)|(cmp12==False)|(cmp13==False)|(cmp14==False)|(cmp15==False)|(cmp16==False)|(cmp17==False)|(cmp18==False)|(cmp19==False)|(cmp20==False)|(cmp21==False):
            print(result1[i]+result2[i]+result3[i]+result4[i]+result5[i]+result6[i]+result7[i])
            if cmp1:
                vote1 += 1
                vote2 += 1
            if cmp2:
                vote1 += 1
                vote3 += 1
            if cmp3:
                vote1 += 1
                vote4 += 1
            if cmp4:
                vote1 += 1
                vote5 += 1
            if cmp5:
                vote1 += 1
                vote6 += 1
            if cmp6:
                vote1 += 1
                vote7 += 1
            if cmp7:
                vote2 += 1
                vote3 += 1
            if cmp8:
                vote2 += 1
                vote4 += 1
            if cmp9:
                vote2 += 1
                vote5 += 1
            if cmp10:
                vote2 += 1
                vote6 += 1
            if cmp11:
                vote2 += 1
                vote7 += 1
            if cmp12:
                vote3 += 1
                vote4 += 1
            if cmp13:
                vote3 += 1
                vote5 += 1
            if cmp14:
                vote3 += 1
                vote6 += 1
            if cmp15:
                vote3 += 1
                vote7 += 1
            if cmp16:
                vote4 += 1
                vote5 += 1
            if cmp17:
                vote4 += 1
                vote6 += 1
            if cmp18:
                vote4 += 1
                vote7 += 1
            if cmp19:
                vote5 += 1
                vote6 += 1
            if cmp20:
                vote5 += 1
                vote7 += 1
            if cmp21:
                vote6 += 1
                vote7 += 1
            print('vote1=', vote1)
            print('vote2=', vote2)
            print('vote3=', vote3)
            print('vote4=', vote4)
            print('vote5=', vote5)
            print('vote6=', vote6)
            print('vote7=', vote7)
            vote = []
            vote.append(vote1)
            vote.append(vote2)
            vote.append(vote3)
            vote.append(vote4)
            vote.append(vote5)
            vote.append(vote6)
            vote.append(vote7)
            #print(max(vote))
            num = vote.index(max(vote))+1
            if num == 1:
                print('set1:', result1[i][0:4]+result1[i][16:])
                f.write(result1[i][0:4]+result1[i][16:])
            elif num == 2:
                print(result2[i])
                print(result2[i][0:4])
                print(result2[i][16:])
                print('set2:', result2[i][0:4]+result2[i][16:])

                f.write(result2[i][0:4] + result2[i][16:])
            elif num == 3:
                print('set3:', result3[i][0:4]+result3[i][16:])
                f.write(result3[i][0:4] + result3[i][16:])
            elif num == 4:
                print('set4:', result4[i][0:4]+result4[i][16:])
                f.write(result4[i][0:4] + result4[i][16:])
            elif num == 5:
                print('set4:', result5[i][0:4]+result5[i][16:])
                f.write(result5[i][0:4] + result5[i][16:])
            elif num == 6:
                print('set4:', result6[i][0:4]+result6[i][16:])
                f.write(result6[i][0:4] + result6[i][16:])
            elif num == 7:
                print('set4:', result7[i][0:4]+result7[i][16:])
                f.write(result7[i][0:4] + result7[i][16:])
            vote1 = 0
            vote2 = 0
            vote3 = 0
            vote4 = 0
            vote5 = 0
            vote6 = 0
            vote7 = 0
            count += 1
        else:
            print('same, set:', result4[0][0:4]+result4[0][16:])
            #print('same, set:', result1[i][0:4] + result1[i][16:])
            f.write(result1[i][0:4] + result1[i][16:])
print('count=', count)
f.close()
f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
f6.close()
f7.close()

