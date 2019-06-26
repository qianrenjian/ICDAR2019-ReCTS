import tensorflow as tf
import os
import random
import tensorflow.contrib.slim as slim
import time
import logging
import numpy as np
import pickle
import linecache

from PIL import Image
from network import *
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

logger = logging.getLogger('Training a chinese write char recognition')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


tf.app.flags.DEFINE_boolean('random_flip_up_down', False, "Whether to random flip up down")
tf.app.flags.DEFINE_boolean('random_brightness', True, "whether to adjust brightness")
tf.app.flags.DEFINE_boolean('random_contrast', True, "whether to random constrast")

tf.app.flags.DEFINE_integer('charset_size', 2, "Choose the first `charset_size` character to conduct our experiment.")
tf.app.flags.DEFINE_integer('image_size', 112, "Needs to provide same value as in training.")
tf.app.flags.DEFINE_boolean('gray', True, "whether to change the rbg to gray")
tf.app.flags.DEFINE_integer('max_steps', 1000, 'the max training steps ')
tf.app.flags.DEFINE_integer('eval_steps', 22, "the step num to eval")
tf.app.flags.DEFINE_integer('save_steps', 22, "the steps to save")
tf.app.flags.DEFINE_integer('decay_step', 22*2, 'Number of decay-step')
tf.app.flags.DEFINE_integer('val_step', 22, 'Number of decay-step')

tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint2/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir', 'H:/why_workspace/ReCTS/img_dir2/', 'the train dataset dir')
tf.app.flags.DEFINE_string('val_data_dir', 'H:/why_workspace/ReCTS/img_dir2/', 'the train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir', 'H:/why_workspace/ReCTS/Task1/img/', 'the test dataset dir')
tf.app.flags.DEFINE_string('label_dir', './label.txt', '')
tf.app.flags.DEFINE_string('result_dir', './result.txt', '')

tf.app.flags.DEFINE_string('log_dir', './log2', 'the logging dir')
tf.app.flags.DEFINE_boolean('restore', True, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_integer('epoch', 10, 'Number of epoches')
tf.app.flags.DEFINE_integer('batch_size', 256, 'Validation batch size')
tf.app.flags.DEFINE_string('mode', 'train', 'Running mode. One of {"train", "validation", "test"}')
FLAGS = tf.app.flags.FLAGS

class DataIterator:
    def __init__(self, data_dir):
        # Set FLAGS.charset_size to a small value if available computation power is limited.
        truncate_path = data_dir + ('%04d' % FLAGS.charset_size)
        print(truncate_path)
        self.image_names = []
        for root, sub_folder, file_list in os.walk(data_dir):
            #print(sub_folder)
            # for file_path in file_list:
            #     print(file_path)
            if root < truncate_path:
                self.image_names += [os.path.join(root, file_path) for file_path in file_list]
                #print(self.image_names)
        random.shuffle(self.image_names)
        self.labels = [int(file_name[len(data_dir):].split(os.sep)[0]) for file_name in self.image_names]

    @property
    def size(self):
        return len(self.labels)

    @staticmethod
    def data_augmentation(images):
        if FLAGS.random_flip_up_down:
            images = tf.image.random_flip_up_down(images)
        if FLAGS.random_brightness:
            images = tf.image.random_brightness(images, max_delta=0.3)
        if FLAGS.random_contrast:
            images = tf.image.random_contrast(images, 0.8, 1.2)
        return images

    def input_pipeline(self, batch_size, num_epochs=None, aug=False):
        images_tensor = tf.convert_to_tensor(self.image_names, dtype=tf.string)
        labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int64)
        input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor], num_epochs=num_epochs)

        labels = input_queue[1]
        images_content = tf.read_file(input_queue[0])
        images = tf.image.convert_image_dtype(tf.image.decode_jpeg(images_content, channels=1), tf.float32)
        if aug:
            images = self.data_augmentation(images)
        new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
        images = tf.image.resize_images(images, new_size)
        image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=50000,
                                                          min_after_dequeue=10000)
        return image_batch, label_batch

class DataIterator_val:
    def __init__(self, data_dir):
        # Set FLAGS.charset_size to a small value if available computation power is limited.
        truncate_path = data_dir + ('%04d' % FLAGS.charset_size)
        print(truncate_path)
        self.image_names = []
        for root, sub_folder, file_list in os.walk(data_dir):
            #print(sub_folder)
            # for file_path in file_list:
            #     print(file_path)
            if root < truncate_path:
                self.image_names += [os.path.join(root, file_path) for file_path in file_list]
                #print(self.image_names)
        #random.shuffle(self.image_names)
        self.labels = [int(file_name[len(data_dir):].split(os.sep)[0]) for file_name in self.image_names]

    @property
    def size(self):
        return len(self.labels)

    @staticmethod
    def data_augmentation(images):
        if FLAGS.random_flip_up_down:
            images = tf.image.random_flip_up_down(images)
        if FLAGS.random_brightness:
            images = tf.image.random_brightness(images, max_delta=0.3)
        if FLAGS.random_contrast:
            images = tf.image.random_contrast(images, 0.8, 1.2)
        return images

    def input_pipeline(self, batch_size, num_epochs=None, aug=False):
        images_tensor = tf.convert_to_tensor(self.image_names, dtype=tf.string)
        labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int64)
        input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor], num_epochs=num_epochs)

        labels = input_queue[1]
        images_content = tf.read_file(input_queue[0])
        images = tf.image.convert_image_dtype(tf.image.decode_jpeg(images_content, channels=1), tf.float32)
        if aug:
            images = self.data_augmentation(images)
        new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
        images = tf.image.resize_images(images, new_size)
        # image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=50000,
        #                                                   min_after_dequeue=10000)
        image_batch, label_batch = tf.train.batch([images, labels], batch_size=batch_size)
        return image_batch, label_batch

class DataIterator_test:
    def __init__(self, data_dir):
        self.image_names = []
        for root, dirs, files in os.walk(FLAGS.test_data_dir):
            for file in files:
                self.image_names += [os.path.join(root, file)]
        #print(self.image_names)
        #random.shuffle(self.image_names)

    def input_pipeline(self, batch_size, num_epochs=None):
        images_tensor = tf.convert_to_tensor(self.image_names, dtype=tf.string)
        input_queue = tf.train.slice_input_producer([images_tensor], num_epochs=num_epochs, shuffle=False)
        images_content = tf.read_file(input_queue[0])
        images = tf.image.convert_image_dtype(tf.image.decode_png(images_content, channels=1), tf.float32)
        new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
        images = tf.image.resize_images(images, new_size)
        # image_batch = tf.train.shuffle_batch([images], batch_size=batch_size, capacity=50000,
        #                                                   min_after_dequeue=10000)
        #images = tf.train.batch([images], batch_size = batch_size)
        return images

def build_graph(top_k, is_train=False, num_classes=FLAGS.charset_size, is_test=False):
    images = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, 1], name='image_batch')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    if is_train:
        net, endpoints = alexnet_v2(images, num_classes=num_classes, is_training=True, reuse=False, dropout_keep_prob=keep_prob)
        pre_label = tf.argmax(net, 1)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net, labels=labels))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net, 1), labels), tf.float32))
        probabilities = tf.nn.softmax(net)
        predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
        accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))
    else:
        if is_test:
            vali_net, vali_end_points = alexnet_v2(images, num_classes=num_classes, is_training=False, reuse=False,
                                                   dropout_keep_prob=keep_prob)
        else:
            vali_net, vali_end_points = alexnet_v2(images, num_classes=num_classes, is_training=False, reuse=True, dropout_keep_prob=keep_prob)
        vali_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=vali_net, labels=labels))
        vali_pre_label = tf.argmax(vali_net, 1)
        vali_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(vali_net, 1), labels), tf.float32))
        vali_probabilities = tf.nn.softmax(vali_net)
        vali_predicted_val_top_k, vali_predicted_index_top_k = tf.nn.top_k(vali_probabilities, k=top_k)
        vali_accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(vali_probabilities, labels, top_k), tf.float32))
        return {'images': images,
                'labels': labels,
                'logits': vali_net,
                'top_k': top_k,
                'loss': vali_loss,
                'accuracy': vali_accuracy,
                'dropout_keep_prob': keep_prob,
                'predicted': vali_pre_label,
                'accuracy_top_k': vali_accuracy_in_top_k,
                'predicted_distribution': vali_probabilities,
                'predicted_index_top_k': vali_predicted_index_top_k,
                'predicted_val_top_k': vali_predicted_val_top_k}
    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    #rate = tf.train.exponential_decay(0.001, global_step, decay_steps=FLAGS.decay_step, decay_rate=0.97, staircase=True)
    opt = tf.train.AdamOptimizer(learning_rate=1e-4)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies([tf.group(*update_ops)]):
        train_op = opt.minimize(loss, global_step=global_step)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    return {'images': images,
            'labels': labels,
            'logits': net,
            'top_k': top_k,
            'global_step': global_step,
            'train_op': train_op,
            'loss': loss,
            'accuracy': accuracy,
            'dropout_keep_prob': keep_prob,
            'accuracy_top_k': accuracy_in_top_k,
            'merged_summary_op': merged_summary_op,
            'predicted': pre_label,
            'predicted_distribution': probabilities,
            'predicted_index_top_k': predicted_index_top_k,
            'predicted_val_top_k': predicted_val_top_k}


def train():
    print('Begin training')
    train_feeder = DataIterator(data_dir=FLAGS.train_data_dir)
    test_feeder = DataIterator_val(data_dir=FLAGS.train_data_dir)
    with tf.Session() as sess:
        train_images, train_labels = train_feeder.input_pipeline(batch_size=FLAGS.batch_size, aug=True, num_epochs=FLAGS.epoch)
        # test_images, test_labels = test_feeder.input_pipeline(batch_size=FLAGS.batch_size, aug=False, num_epochs = FLAGS.epoch+2)
        graph = build_graph(top_k=1, is_train=True)
        vali_graph = build_graph(top_k=1, is_train=False, is_test=False)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        start_step = 0
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print("restore from the checkpoint {0}".format(ckpt))
                start_step += int(ckpt.split('-')[-1])
        logger.info(':::Training Start:::')
        try:
            best_acc = 0.0
            num_count = -1
            num_epoch = 0
            while not coord.should_stop():
                train_images_batch, train_labels_batch = sess.run([train_images, train_labels])
                feed_dict = {graph['images']: train_images_batch,
                             graph['labels']: train_labels_batch,
                             graph['dropout_keep_prob']: 0.9}
                _, loss_val, train_summary, step = sess.run(
                    [graph['train_op'], graph['loss'], graph['merged_summary_op'], graph['global_step']],
                    feed_dict=feed_dict)
                train_writer.add_summary(train_summary, step)
                #logger.info("the step {0} loss {1}".format(step, loss_val))
                if step > FLAGS.max_steps:
                    logger.info('epoch {0} best accuracy {1}'.format(epoch_num, best_acc))
                    break
                if num_count == 3:
                    break
                if step % FLAGS.eval_steps == 1:
                    coord2 = tf.train.Coordinator()
                    threads2 = tf.train.start_queue_runners(sess=sess, coord=coord2)
                    try:
                        test_images, test_labels = test_feeder.input_pipeline(batch_size=FLAGS.batch_size, aug=False,
                                                                              num_epochs=1)
                        i = 0
                        print(111)
                        acc_top_1, acc_top_k = 0.0, 0.0
                        while not coord2.should_stop():
                            i += 1
                            #start_time = time.time()
                            print(222)
                            test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
                            print(333)
                            feed_dict = {vali_graph['images']: test_images_batch,
                                         vali_graph['labels']: test_labels_batch,
                                         vali_graph['dropout_keep_prob']: 1.0}
                            acc_1 = sess.run(
                                vali_graph['accuracy'],
                                feed_dict=feed_dict)
                            acc_top_1 += acc_1
                            #end_time = time.time()
                            # logger.info("the batch {0} takes {1} seconds, accuracy = {2}(top_1)"
                            #             .format(i, end_time - start_time, acc_1))
                            if i == FLAGS.val_step+1:
                                acc_top_1 = acc_top_1 / i
                                if num_count == -1:
                                    best_acc = acc_top_1
                                    num_count+=1
                                if acc_top_1 < best_acc:
                                    num_count+=1
                                    print('count=', num_count)
                                else:
                                    best_acc = acc_top_1
                                    print('↑')
                                    num_count=0
                                    logger.info('Save the ckpt of {0}'.format(step))
                                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'),
                                               global_step=graph['global_step'])
                                logger.info('epoch {0} top 1 accuracy {1} ,best accuracy {2}'.format(num_epoch ,acc_top_1, best_acc))
                                num_epoch+=1
                                break
                    except tf.errors.OutOfRangeError:
                        logger.info('==================Validation Finished================')
                        acc_top_1 = acc_top_1 / (i - 1)
                        logger.info('top 1 accuracy {0}'.format(acc_top_1))
                    finally:
                        coord2.request_stop()
                    coord2.join(threads2)

        except tf.errors.OutOfRangeError:
            logger.info('==================Train Finished================')
            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'), global_step=graph['global_step'])
        finally:
            coord.request_stop()
        coord.join(threads)

def inference(image, is_test = False):
    temp_image = Image.open(image).convert('L')
    temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
    temp_image = np.asarray(temp_image) / 255.0
    temp_image = temp_image.reshape([-1, 112, 112, 1])
    with tf.Session() as sess:
        #logger.info('========start inference============')
        graph = build_graph(top_k=1, is_train=False, is_test=is_test)
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
        predict_val, predict_index = sess.run([graph['predicted_val_top_k'], graph['predicted_index_top_k']],
                                              feed_dict={graph['images']: temp_image, graph['dropout_keep_prob']: 1.0})
    return predict_val, predict_index

def validation_output_txt():
    test_feeder = DataIterator_test(data_dir=FLAGS.test_data_dir)
    final_predict_val = []
    final_predict_index = []
    with tf.Session() as sess:
        test_images = test_feeder.input_pipeline(batch_size=FLAGS.batch_size, num_epochs=1)
        graph = build_graph(top_k=1, is_train=False, is_test=True)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())  # initialize test_feeder's inside state
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print("restore from the checkpoint {0}".format(ckpt))
        logger.info(':::Start validation:::')
        try:
            i = 0
            while not coord.should_stop():
                i += 1
                test_images_batch = sess.run([test_images])
                feed_dict = {graph['images']: test_images_batch,
                             graph['dropout_keep_prob']: 1.0}
                probs, indices = sess.run([graph['predicted_val_top_k'],
                                           graph['predicted_index_top_k']], feed_dict=feed_dict)

                final_predict_val += probs.tolist()
                final_predict_index += indices.tolist()
        except tf.errors.OutOfRangeError:
            count = 0
            for root, dirs, files in os.walk(FLAGS.test_data_dir):
                for file in files:
                    the_line = linecache.getline(FLAGS.label_dir, final_predict_index[count][0] + 1)
                    txt_path = FLAGS.result_dir
                    with open(txt_path, "a", encoding='utf-8') as f:
                        f.write(file + ',' + the_line)
                    count += 1
            logger.info('==================Validation Finished================')
        finally:
            coord.request_stop()
        coord.join(threads)
    return {'prob': final_predict_val, 'indices': final_predict_index}

def validation():
    print('validation')
    test_feeder = DataIterator_val(data_dir=FLAGS.val_data_dir)
    final_predict_val = []
    final_predict_index = []
    groundtruth = []
    with tf.Session() as sess:
        test_images, test_labels = test_feeder.input_pipeline(batch_size=FLAGS.batch_size, num_epochs=1)
        graph = build_graph(top_k=1, is_train=False, is_test=True)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())  # initialize test_feeder's inside state
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print("restore from the checkpoint {0}".format(ckpt))
        logger.info(':::Start validation:::')
        try:
            i = 0
            acc_top_1, acc_top_k = 0.0, 0.0
            while not coord.should_stop():
                i += 1
                start_time = time.time()
                test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
                feed_dict = {graph['images']: test_images_batch,
                             graph['labels']: test_labels_batch,
                             graph['dropout_keep_prob']: 1.0}
                acc_1 = sess.run(
                    graph['accuracy'],
                    feed_dict=feed_dict)
                acc_top_1 += acc_1
                end_time = time.time()
                logger.info("the batch {0} takes {1} seconds, accuracy = {2}(top_1)"
                            .format(i, end_time - start_time, acc_1))
        except tf.errors.OutOfRangeError:
            logger.info('==================Validation Finished================')
            acc_top_1 = acc_top_1 / (i-1)
            logger.info('top 1 accuracy {0}'.format(acc_top_1))
        finally:
            coord.request_stop()
        coord.join(threads)
    return {'prob': final_predict_val, 'indices': final_predict_index, 'groundtruth': groundtruth}

def main(_):
    if FLAGS.mode == "train":
        train()
    elif FLAGS.mode == "validation_output_txt":
        validation_output_txt()
    elif FLAGS.mode == "validation":
        validation()
    elif FLAGS.mode == 'inference':
        count = 0
        for root, dirs, files in os.walk(FLAGS.test_data_dir):
            for file in files:
                image_path = FLAGS.test_data_dir + file
                if count == 0:
                    final_predict_val, final_predict_index = inference(image_path, is_test=True)
                else:
                    final_predict_val, final_predict_index = inference(image_path)
                count += 1
                the_line = linecache.getline(FLAGS.label_dir, final_predict_index[0][0]+1)
                txt_path = FLAGS.result_dir
                with open(txt_path, "wb", encoding='utf-8') as f:
                    f.write(file+ ',' + the_line)

if __name__ == "__main__":
    tf.app.run()