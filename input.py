# -*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# 训练集的路径
train_file = '/media/jxnu/Files/dog_vs_cat/train'

def get_file(file_path):

    cats = []
    label_cats = []
    dogs = []
    label_dogs = []

    for file in os.listdir(file_path):
        # 遍历文件夹里的所有照片
        cat = 'cat'
        if cat in file:
            cats.append(file_path + '/' + file)
            # 找出猫图
            label_cats.append(0)
            # 猫的标签为0
        else :
            dogs.append(file_path + '/' + file)
            label_dogs.append(1)

    # print "共有%d 只猫, %d 只狗"%(len(label_cats), len(label_dogs))

    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))
    # print image_list


    temp = np.array([image_list, label_list])
    # 使用array 来打乱顺序


    # print temp
    temp = temp.transpose()
    # 纵向
    # print temp
    np.random.shuffle(temp)
    # print temp
    # 打乱顺序


    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]


    return image_list, label_list

def get_batch(image, label, image_w, image_h, batch_size, capacity):

    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    # 类型转换


    input_queue = tf.train.slice_input_producer([image, label])
    # 输入的队列, 生成队列函数,

    label = input_queue[1]
    image_content = tf.read_file(input_queue[0])
    # 读取图片文件

    image = tf.image.decode_jpeg(image_content, channels=3)
    # 图片解码成jpg图片格式,通道3 彩色图片

    image = tf.image.resize_image_with_crop_or_pad(image, image_w,
                                                   image_h)
    # reshape 图片, 裁剪或填充 ,从中心开始的

    # image = tf.image.per_image_standardization(image)

    # 数据标准化,

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)

    image_batch = tf.cast(image_batch, tf.float32)
    label_batch = tf.reshape(label_batch, [batch_size])


    return image_batch, label_batch




# 测试batch 函数是否工作

batch_size = 1
capacity = 256
image_h = 227
image_w = 227

image_list, label_list = get_file(train_file)
image_batch, label_batch = get_batch(image_list, label_list,
                                     image_w=image_w,image_h=image_h,
                                     batch_size=batch_size,
                                     capacity=capacity)


# with tf.Session() as sess:
#
#     t = 0
#     coord = tf.train.Coordinator()
#     # 用来帮助多个线程协同合作,多个线程同步终止
#
#     threads = tf.train.start_queue_runners(coord=coord)
#     #  线程开启,使用队列
#
#     try:
#         while not coord.should_stop() and t < 3:
#             img, label = sess.run([image_batch, label_batch])
#             convert = tf.cast(img, tf.float32)
#             sess.run(convert)
#
#             for i in np.arange(batch_size):
#                 print('label : %d'%label[i])
#                 plt.imshow(img[i, :, :, :])
#                 plt.show()
#
#             t += 1
#
#     except tf.errors.OutOfRangeError:
#     #　防止错误使程序结束
#     # 捕捉异常
#         print "done"
#     finally:
#         coord.request_stop()
#     # 请求停止
#     coord.join(threads)
#     #等待被指定的 线程终止



















