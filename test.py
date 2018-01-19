# -*- coding:utf-8 -*-
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import input
import model


def get_one_image(train):
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]
# 随机抽取一张图片
    image = Image.open(img_dir)
    plt.imshow(image)
    plt.show()

    image = image.resize([208, 208])
    image = np.array(image)
    return image


def evaluate_one_image():
    train_file = '/media/jxnu/Files/dog_vs_cat/train'
    train, train_label = input.get_file(train_file)

    image_array = get_one_image(train)

    # with tf.Graph.as_default():
    batch_size = 1
    n_classes = 2

    image = tf.cast(image_array, tf.float32)
    image = tf.reshape(image, [1, 208, 208, 3])
    logit = model.inference(image, batch_size, n_classes)
    logit = tf.nn.softmax(logit)

    x = tf.placeholder(tf.float32, shape=[208, 208, 3])

    logs_train_dir = '/media/jxnu/Files/PycharmProjects/tf1/dogs_vs_cats/train_logs'
    saver = tf.train.Saver()

    with tf.Session() as sess:
        print('reading checkpoints...')

        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('.')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('loading success, global_step is %s'%global_step)

        else:
            print('No checkpoint file found ')

        prediction = sess.run(logit, feed_dict={x: image_array})
        max_index = np.argmax(prediction)

        if max_index == 0:
                print('this is a cat with possibility %.6f'%prediction[:, 0])
        else:
                print('this is a dog with possibility %.6f' % prediction[:, 1])




evaluate_one_image()