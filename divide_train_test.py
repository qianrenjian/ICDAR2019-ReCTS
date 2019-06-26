import tensorflow as tf
import glob
import os.path
import time
from shutil import copy2

# 原始输入数据的目录，这个目录下有五个子目录，每个子目录底下保存属于该类别的所有图片
INPUT_DATA = 'H:/why_workspace/ReCTS/img_dir2/'

TRAIN_OUTPUT = 'H:/why_workspace/ReCTS/img_dir3/train/'
TEST_OUTPUT = 'H:/why_workspace/ReCTS/img_dir3/validation/'


# 读取数据并将数据分割成训练数据测试数据(未处理的图像地址list)
def cpy_image_lists(sess):
    sub_dirs = []
    sub_file = os.listdir(INPUT_DATA)
    img = []
    for dirs in sub_file:
        dst_train = os.path.join(TRAIN_OUTPUT, dirs)
        dst_test = os.path.join(TEST_OUTPUT, dirs)
        os.mkdir(dst_train)
        os.mkdir(dst_test)
        img = os.listdir(os.path.join(INPUT_DATA, dirs))
        num_pic = len(img)
        num_train = int(0.8*num_pic)
        if num_train == 0:
            src = os.path.join(os.path.join(INPUT_DATA, dirs), img[0])
            copy2(src, dst_train)
            src = os.path.join(os.path.join(INPUT_DATA, dirs), img[0])
            copy2(src, dst_test)
        else:
            for i in range(0, num_train):
                src = os.path.join(os.path.join(INPUT_DATA, dirs), img[i])
                copy2(src, dst_train)
            for i in range(num_train, num_pic):
                src = os.path.join(os.path.join(INPUT_DATA, dirs), img[i])
                copy2(src, dst_test)
        img = []

    # for dirs in sub_file:
    #     os.mkdir(os.path.join(TRAIN_OUTPUT, dirs))
    #     #print(dirs)
    #     m = os.path.join(INPUT_DATA, dirs)
    #     if os.path.isdir(m):
    #         sub_dirs.append(m)
    # for sub_dir in sub_dirs:
    #     next_sub_dirs = os.listdir(sub_dir)
    #     #print('next_sub_dirs is ', next_sub_dirs)
    #     # 获取一个子目录中所有的图片文件
    #     for next_sub_dir in next_sub_dirs:
    #         file_list = []
    #         file_glob = os.path.join(sub_dir, next_sub_dir)
    #         print(file_glob)
    #         #print(file_glob[32:36])
    #         file_list.extend(file_glob)
    #         num_example = len(file_list)
    #         num_train = int(num_example * 0.8)
    #         #print(file_list)
    #         print(num_example)
            #print(file_list[1])
            # for i in range(0, num_train):
            #     print(file_list[i])
                #copy2(file_list[i])
            # #for extension in extensions:
            #     # print('*****当前图片后缀为：', extension, " *****")
            #     # os.path.join()合成地址
            #     file_glob = os.path.join(sub_dir, next_sub_dir)
            #     #print(file_glob)
            #     # print("*****当前图片地址格式为：", file_glob, " *****")
            #     # glob.glob（）搜索所有符合要求的文件
            #     file_list.extend(glob.glob(file_glob))
            #     if not file_list:  # 如果file_list为空list，就判断为False
            #         continue
            #
            #     # 处理图片数据
            #     num_example = len(file_list)
            #     num_train = int(num_example * 0.8)
            #     # print("*****当前file_list的图片数量为：", num_example, "*****")
            #     # 创建目录文件夹
            #     #os.mkdir(os.path.join(TRAIN_OUTPUT, next_sub_dir))
            #     #print(os.path.join(TRAIN_OUTPUT, next_sub_dir))
            #     #os.mkdir(os.path.join(TEST_OUTPUT, next_sub_dir))
            #     #print(os.path.join(TEST_OUTPUT, next_sub_dir))
            #     for i in range(0, num_train):
            #         png_name1 = os.path.basename(file_list[i])
            #         train_output1 = os.path.join(TRAIN_OUTPUT, next_sub_dir, png_name1)
            #         #print(file_list[i])
            #         #print(train_output1)
            #         #copy2(file_list[i], train_output1)
            #
            #     for n in range(num_example - num_train):
            #         png_name2 = os.path.basename(file_list[n + num_train])
            #         train_output2 = os.path.join(TEST_OUTPUT, next_sub_dir, png_name2)
            #         #copy2(file_list[n + num_train], train_output2)
            # end_time = time.time()
            # #print("the folder {0} takes time is {1}".format(next_sub_dir, end_time - start_time))



def main(argv=None):
    config = tf.ConfigProto(allow_soft_placement=True)
    # 设置根据需求增长使用的内存
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        cpy_image_lists(sess)
    print("all finished")

if __name__ == '__main__':
    tf.app.run()