import tensorflow as tf
import numpy as np
import os
import input
import model

n_classes = 2
image_w = 208
image_h = 208
learning_rate = 0.0001
capacity = 2000
batch_size = 16
max_step = 12000

def run_model():
    train_file = '/media/jxnu/Files/dog_vs_cat/train'
    train_logs = '/media/jxnu/Files/PycharmProjects/tf1/dogs_vs_cats/train_logs'

    train, train_labels = input.get_file(train_file)

    train_batch, train_label_batch = input.get_batch(train,
                                                     train_labels,
                                                     image_w,
                                                     image_h,
                                                     batch_size,
                                                     capacity)

    train_logits = model.inference(train_batch, batch_size, n_classes)
    train_loss = model.losses(train_logits, train_label_batch)
    train_op = model.trainning(train_loss, learning_rate)

    train_acc = model.evaluate(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(train_logs, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(max_step):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])

            if step % 50 == 0:
                print('step %d, train loss: %.2f, train accuracy: %.2f'%(step, tra_loss, tra_acc))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 20000 == 0 or (step+1) == max_step:
                checkpoint_path = os.path.join(train_logs, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print("Done training  -- epoch limited reached")
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


run_model()


