#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Train the CTC model with multiple GPUs (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import sys
import time
from os.path import join, isfile, abspath
from setproctitle import setproctitle

import yaml

sys.path.append(abspath('../../../'))
from examples.aishell.data.load_dataset_ctc import Dataset
from examples.aishell.metrics.ctc import do_eval_cer, do_eval_wer
from utils.io.labels.sparsetensor import list2sparsetensor
from utils.training.learning_rate_controller import Controller
from utils.training.plot import plot_loss, plot_ler
from utils.training.multi_gpu import average_gradients
from utils.directory import mkdir_join, mkdir
from utils.parameter import count_total_parameters
from models.ctc.ctc import CTC
import tensorflow as tf
import collections
import soundfile
import numpy as np
from python_speech_features import delta, fbank


class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "source", "target_input",
                            "target_output", "source_sequence_length",
                            "target_sequence_length", "filename", "duration", "label"))):
    pass


def get_iterator2(
        src_dataset,
        tgt_dataset,
        src_vocab_table,
        tgt_vocab_table,
        batch_size,
        sos,
        eos,
        random_seed,
        num_buckets,
        src_max_len=None,
        tgt_max_len=None,
        num_parallel_calls=4,
        output_buffer_size=None,
        skip_count=None,
        num_shards=1,
        shard_index=0,
        label_dict={}):

    def fbank_from_file(wav_path, duration, label):
        sig1, sr1 = soundfile.read(wav_path, dtype='float32')
        fbank_feat, energy = fbank(sig1, sr1, nfilt=40)  # (407, 40)
        # fbank_feat = np.column_stack(
        #     (np.log(energy), np.log(fbank_feat)))  # (407, 41)
        d_fbank_feat = delta(fbank_feat, 2)
        dd_fbank_feat = delta(d_fbank_feat, 2)
        # print(label_dict)
        # print(np.array([label_dict.get(i) for i in label.strip().split()]))
        label_list = np.array([int(i) for i in label.strip().split()]).astype(np.int32)
        # print(dd_fbank_feat.dtype)
        # concat_fbank_feat = np.array(
        #     [fbank_feat, d_fbank_feat, dd_fbank_feat], dtype=np.float32)  # (3, 407, 41)
        # return concat_fbank_feat, wav_path, duration, label
        return np.column_stack((np.log(fbank_feat), d_fbank_feat, dd_fbank_feat)).astype(np.float32), wav_path, duration, label_list

    # src_dataset = src_dataset.shard(num_shards, shard_index)

    src_dataset = src_dataset.map(lambda filename, duration, label: tuple(tf.py_func(
        fbank_from_file, [filename, duration, label], [
            tf.float32, tf.string, tf.float32, tf.int32]
    )), num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # tf.size(src)
    src_dataset = src_dataset.map(
        lambda src, filename, duration, label: (
            src, tf.shape(src)[0], filename, duration, label)
    ).prefetch(output_buffer_size)

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The first three entries are the source and target line rows;
            # these have unknown-length vectors.  The last two entries are
            # the source and target row sizes; these are scalars.
            padded_shapes=(
                tf.TensorShape([None, None]),  # src
                # tf.TensorShape([None]),  # tgt_input
                # tf.TensorShape([None]),  # tgt_output
                tf.TensorShape([]),  # src_len
                tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape([None])
            ),  # tgt_len
            # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(
                0.,  # src
                # tgt_eos_id,  # tgt_input
                # tgt_eos_id,  # tgt_output
                # 0,  # src_len -- unused
                0,
                "",
                0.,
                0))  # tgt_len -- unused

    def key_func(unused_1, src_len, filename, duration, label):
        # Calculate bucket_width by maximum source sequence length.
        # Pairs with length [0, bucket_width) go to bucket 0, length
        # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
        # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
        # if src_max_len:
        #   bucket_width = (src_max_len + num_buckets - 1) // num_buckets
        # else:
        bucket_width = 100
        print("%s:%s" % (filename, src_len))

        # Bucket sentence pairs by the length of their source sentence and target
        # sentence. // 3 // 40
        bucket_id = tf.maximum(src_len // bucket_width, 1)
        return tf.to_int64(tf.minimum(16, bucket_id))

    def reduce_func(unused_key, windowed_data):
        return batching_func(windowed_data)

    batched_dataset = src_dataset.apply(
        tf.contrib.data.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

    batched_dataset = batched_dataset.shuffle(output_buffer_size, random_seed)

    batched_iter = batched_dataset.make_initializable_iterator()
    (src_ids, src_seq_len, filename, duration, label) = (batched_iter.get_next())
    return BatchedInput(
        initializer=batched_iter.initializer,
        source=src_ids,
        target_input=None,
        target_output=None,
        source_sequence_length=src_seq_len,
        target_sequence_length=None,
        filename=filename,
        duration=duration,
        label=label)

def dense_to_sparse(dense_tensor):
    indices = tf.where(tf.not_equal(dense_tensor, tf.constant(0, dense_tensor.dtype)))
    values = tf.gather_nd(dense_tensor, indices)
    shape = tf.shape(dense_tensor, out_type=np.int64)
    return tf.SparseTensor(indices, values, shape)


def do_train(model, params, gpu_indices):
    """Run CTC training.
    Args:
        model: the model to train
        params (dict): A dictionary of parameters
        gpu_indices (list): GPU indices
    """
    
    print(params['train_data_size'])
    # Load dataset
    train_data = Dataset(
        data_type='train', train_data_size=params['train_data_size'],
        label_type=params['label_type'],
        batch_size=params['batch_size'], max_epoch=params['num_epoch'],
        splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=True, sort_stop_epoch=params['sort_stop_epoch'],
        num_gpu=len(gpu_indices), dataset_root=params["dataset_root"],
        json_file_path=params["train_json"], label_dict_file=params["label_dict_file"],
        params=params)
    # Load dev dataset
    dev_data = Dataset(
        data_type='dev', train_data_size=params['train_data_size'],
        label_type=params['label_type'],
        batch_size=params['batch_size'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=True, num_gpu=len(gpu_indices), json_file_path=params["dev_json"],
        label_dict_file=params["label_dict_file"])


    # Tell TensorFlow that the model will be built into the default graph
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data.input_paths,
            train_data.durations, train_data.target_label_indexes))
        batched_iter = get_iterator2(train_dataset, None, None, None, params['batch_size'],
                                    None, None, random_seed=3,
                                    num_buckets=16, output_buffer_size=params['batch_size'] * 2,
                                    num_parallel_calls=4, num_shards=1, shard_index=0)
        source = batched_iter.source
        source.set_shape([None, None, 120])
        src_seq_len = batched_iter.source_sequence_length
        duration = batched_iter.duration
        label_list = batched_iter.label
        filename = batched_iter.filename

        sources_splits = tf.split(axis=0, num_or_size_splits=len(gpu_indices), value=source)
        labels_splits = tf.split(axis=0, num_or_size_splits=len(gpu_indices), value=label_list)
        src_seq_len_splits = tf.split(axis=0, num_or_size_splits=len(gpu_indices), value=src_seq_len)

        # Create a variable to track the global step
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Set optimizer
        learning_rate_pl = tf.placeholder(tf.float32, name='learning_rate')
        optimizer = model._set_optimizer(
            params['optimizer'], learning_rate_pl)

        # Calculate the gradients for each model tower
        total_grads_and_vars, total_losses = [], []
        decode_ops, ler_ops = [], []
        all_devices = ['/gpu:%d' % i_gpu for i_gpu in range(len(gpu_indices))]
        # NOTE: /cpu:0 is prepared for evaluation
        with tf.variable_scope(tf.get_variable_scope()):
            for i_gpu in range(len(all_devices)):
                with tf.device(all_devices[i_gpu]):
                    with tf.name_scope('tower_gpu%d' % i_gpu) as scope:

                        # Define placeholders in each tower
                        model.create_placeholders()
                        
                        tower_loss, tower_logits = model.compute_loss(
                            sources_splits[i_gpu],
                            dense_to_sparse(labels_splits[i_gpu]),
                            src_seq_len_splits[i_gpu],
                            model.keep_prob_pl_list[i_gpu],
                            scope)

                        # Calculate the total loss for the current tower of the
                        # model. This function constructs the entire model but
                        # shares the variables across all towers.
                        # tower_loss, tower_logits = model.compute_loss(
                        #     model.inputs_pl_list[i_gpu],
                        #     model.labels_pl_list[i_gpu],
                        #     model.inputs_seq_len_pl_list[i_gpu],
                        #     model.keep_prob_pl_list[i_gpu],
                        #     scope)
                        tower_loss = tf.expand_dims(tower_loss, axis=0)
                        total_losses.append(tower_loss)

                        # Reuse variables for the next tower
                        tf.get_variable_scope().reuse_variables()

                        # Calculate the gradients for the batch of data on this
                        # tower
                        tower_grads_and_vars = optimizer.compute_gradients(
                            tower_loss)

                        # Gradient clipping
                        tower_grads_and_vars = model._clip_gradients(
                            tower_grads_and_vars)

                        # TODO: Optionally add gradient noise

                        # Keep track of the gradients across all towers
                        total_grads_and_vars.append(tower_grads_and_vars)

                        # Add to the graph each operation per tower
                        decode_op_tower = model.decoder(
                            tower_logits,
                            model.inputs_seq_len_pl_list[i_gpu],
                            beam_width=params['beam_width'])
                        decode_ops.append(decode_op_tower)
                        ler_op_tower = model.compute_ler(
                            decode_op_tower, model.labels_pl_list[i_gpu])
                        ler_op_tower = tf.expand_dims(ler_op_tower, axis=0)
                        ler_ops.append(ler_op_tower)

        # Aggregate losses, then calculate average loss
        total_losses = tf.concat(axis=0, values=total_losses)
        loss_op = tf.reduce_mean(total_losses, axis=0)
        ler_ops = tf.concat(axis=0, values=ler_ops)
        ler_op = tf.reduce_mean(ler_ops, axis=0)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers
        average_grads_and_vars = average_gradients(total_grads_and_vars)

        # Apply the gradients to adjust the shared variables.
        train_op = optimizer.apply_gradients(average_grads_and_vars,
                                             global_step=global_step)

        # Define learning rate controller
        lr_controller = Controller(
            learning_rate_init=params['learning_rate'],
            decay_start_epoch=params['decay_start_epoch'],
            decay_rate=params['decay_rate'],
            decay_patient_epoch=params['decay_patient_epoch'],
            lower_better=True)

        # Build the summary tensor based on the TensorFlow collection of
        # summaries
        summary_train = tf.summary.merge(model.summaries_train)
        summary_dev = tf.summary.merge(model.summaries_dev)

        # Add the variable initializer operation
        init_op = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints
        saver = tf.train.Saver(max_to_keep=None)

        # Count total parameters
        parameters_dict, total_parameters = count_total_parameters(
            tf.trainable_variables())
        for parameter_name in sorted(parameters_dict.keys()):
            print("%s %s" % (parameter_name, parameters_dict[parameter_name]))
        print("Total %d variables, %s M parameters" %
              (len(parameters_dict.keys()),
               "{:,}".format(total_parameters / 1000000)))

        csv_steps, csv_loss_train, csv_loss_dev = [], [], []
        csv_ler_train, csv_ler_dev = [], []
        # Create a session for running operation on the graph
        # NOTE: Start running operations on the Graph. allow_soft_placement
        # must be set to True to build towers on GPU, as some of the ops do not
        # have GPU implementations.
        config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
        config.gpu_options.allow_growth = True
        # import pdb;pdb.set_trace()
        with tf.Session(config=config) as sess:

            # Instantiate a SummaryWriter to output summaries and the graph
            summary_writer = tf.summary.FileWriter(
                model.save_path, sess.graph)

            # Initialize parameters
            sess.run(init_op)
            

            # Train model
            start_time_train = time.time()
            start_time_epoch = time.time()
            start_time_step = time.time()
            ler_dev_best = 1
            not_improved_epoch = 0
            learning_rate = float(params['learning_rate'])
            # for step, (data, is_new_epoch) in enumerate(train_data):
            for epoch in range(2):
                sess.run(batched_iter.initializer)
                step = 0
                while True:
                    try:
                        is_new_epoch = False
                        # data, is_new_epoch = train_data.next()
                        # Create feed dictionary for next mini batch (train)
                        # inputs, labels, inputs_seq_len, _ = data
                        # labels3 = list2sparsetensor(labels[0], padded_value=train_data.padded_value)
                        # inputs2, labels2, inputs_seq_len2 = sess.run([sources_splits[0],
                        #             dense_to_sparse(labels_splits[0]),
                        #             src_seq_len_splits[0]])
                        feed_dict_train = {}
                        for i_gpu in range(len(gpu_indices)):
                            # feed_dict_train[model.inputs_pl_list[i_gpu]
                            #                 ] = inputs[i_gpu]
                            # feed_dict_train[model.labels_pl_list[i_gpu]] = list2sparsetensor(
                            #     labels[i_gpu], padded_value=train_data.padded_value)
                            # feed_dict_train[model.inputs_seq_len_pl_list[i_gpu]
                            #                 ] = inputs_seq_len[i_gpu]
                            feed_dict_train[model.keep_prob_pl_list[i_gpu]
                                            ] = 1 - float(params['dropout'])
                        feed_dict_train[learning_rate_pl] = learning_rate

                        # Update parameters
                        _, loss_train = sess.run([train_op, loss_op], feed_dict=feed_dict_train)
                        print("epoch: %s, step: %s, loss_train: %s" % (epoch, step, loss_train))
                        step += 1
                    except tf.errors.OutOfRangeError:
                        break

                # # if (step + 1) % int(params['print_step'] / len(gpu_indices)) == 0:
                #     # import pdb;pdb.set_trace()
                #     # Create feed dictionary for next mini batch (dev)
                #     inputs, labels, inputs_seq_len, _ = dev_data.next()[0]
                #     feed_dict_dev = {}
                #     for i_gpu in range(len(gpu_indices)):
                #         feed_dict_dev[model.inputs_pl_list[i_gpu]
                #                       ] = inputs[i_gpu]
                #         feed_dict_dev[model.labels_pl_list[i_gpu]] = list2sparsetensor(
                #             labels[i_gpu], padded_value=dev_data.padded_value)
                #         feed_dict_dev[model.inputs_seq_len_pl_list[i_gpu]
                #                       ] = inputs_seq_len[i_gpu]
                #         feed_dict_dev[model.keep_prob_pl_list[i_gpu]] = 1.0

                #     # Compute loss
                #     loss_train = sess.run(loss_op, feed_dict=feed_dict_train)
                #     loss_dev = sess.run(loss_op, feed_dict=feed_dict_dev)
                #     csv_steps.append(step)
                #     csv_loss_train.append(loss_train)
                #     csv_loss_dev.append(loss_dev)

                #     # Change to evaluation mode
                #     for i_gpu in range(len(gpu_indices)):
                #         feed_dict_train[model.keep_prob_pl_list[i_gpu]] = 1.0

                #     # Compute accuracy & update event files
                #     # ler_train, summary_str_train = sess.run(
                #     #     [ler_op, summary_train], feed_dict=feed_dict_train)
                #     # ler_dev, summary_str_dev = sess.run(
                #     #     [ler_op, summary_dev], feed_dict=feed_dict_dev)
                #     # csv_ler_train.append(ler_train)
                #     # csv_ler_dev.append(ler_dev)
                #     # summary_writer.add_summary(summary_str_train, step + 1)
                #     # summary_writer.add_summary(summary_str_dev, step + 1)
                #     summary_writer.flush()

                #     duration_step = time.time() - start_time_step
                #     # print("Step %d (epoch: %.3f): loss = %.3f (%.3f) / ler = %.3f (%.3f) / lr = %.5f (%.3f min)" %
                #     #       (step + 1, train_data.epoch_detail, loss_train, loss_dev, ler_train, ler_dev,
                #     #        learning_rate, duration_step / 60))
                #     print("Step %d (epoch: %.3f): loss = %.3f (%.3f) / lr = %.5f (%.3f min)" %
                #           (step + 1, train_data.epoch_detail, loss_train, loss_dev,
                #            learning_rate, duration_step / 60))
                #     sys.stdout.flush()
                #     start_time_step = time.time()

                # Save checkpoint and evaluate model per epoch
                # # if is_new_epoch:
                #     duration_epoch = time.time() - start_time_epoch
                #     print('-----EPOCH:%d (%.3f min)-----' %
                #           (train_data.epoch, duration_epoch / 60))

                #     # Save fugure of loss & ler
                #     plot_loss(csv_loss_train, csv_loss_dev, csv_steps,
                #               save_path=model.save_path)
                #     # plot_ler(csv_ler_train, csv_ler_dev, csv_steps,
                #     #          label_type=params['label_type'],
                #     #          save_path=model.save_path)

                #     if train_data.epoch >= params['eval_start_epoch']:
                #         start_time_eval = time.time()
                #         if 'char' in params['label_type']:
                #             print('=== Dev Data Evaluation ===')
                #             # dev-clean
                #             cer_dev_clean_epoch, wer_dev_clean_epoch = do_eval_cer(
                #                 session=sess,
                #                 decode_ops=decode_ops,
                #                 model=model,
                #                 dataset=dev_data,
                #                 label_type=params['label_type'],
                #                 eval_batch_size=1)
                #             print('  CER (clean): %f %%' %
                #                   (cer_dev_clean_epoch * 100))
                #             print('  WER (clean): %f %%' %
                #                   (wer_dev_clean_epoch * 100))

                #             metric_epoch = cer_dev_clean_epoch

                #             if metric_epoch < ler_dev_best:
                #                 ler_dev_best = metric_epoch
                #                 not_improved_epoch = 0
                #                 print('■■■ ↑Best Score (CER)↑ ■■■')

                #                 # Save model (check point)
                #                 checkpoint_file = join(
                #                     model.save_path, 'model.ckpt')
                #                 save_path = saver.save(
                #                     sess, checkpoint_file, global_step=train_data.epoch)
                #                 print("Model saved in file: %s" % save_path)

                #                 # print('=== Test Data Evaluation ===')
                #                 # test-clean
                #                 # cer_test_clean_epoch, wer_test_clean_epoch = do_eval_cer(
                #                 #     session=sess,
                #                 #     decode_ops=decode_ops,
                #                 #     model=model,
                #                 #     dataset=test_clean_data,
                #                 #     label_type=params['label_type'],
                #                 #     is_test=True,
                #                 #     eval_batch_size=1)
                #                 # print('  CER (clean): %f %%' %
                #                 #       (cer_test_clean_epoch * 100))
                #                 # print('  WER (clean): %f %%' %
                #                 #       (wer_test_clean_epoch * 100))
                #                 #
                #                 # # test-other
                #                 # cer_test_other_epoch, wer_test_other_epoch = do_eval_cer(
                #                 #     session=sess,
                #                 #     decode_ops=decode_ops,
                #                 #     model=model,
                #                 #     dataset=test_other_data,
                #                 #     label_type=params['label_type'],
                #                 #     is_test=True,
                #                 #     eval_batch_size=1)
                #                 # print('  CER (other): %f %%' %
                #                 #       (cer_test_other_epoch * 100))
                #                 # print('  WER (other): %f %%' %
                #                 #       (wer_test_other_epoch * 100))
                #             else:
                #                 not_improved_epoch += 1

                #         else:
                #             print('=== Dev Data Evaluation ===')
                #             # dev-clean
                #             wer_dev_clean_epoch = do_eval_wer(
                #                 session=sess,
                #                 decode_ops=decode_ops,
                #                 model=model,
                #                 dataset=dev_data,
                #                 train_data_size=params['train_data_size'],
                #                 eval_batch_size=1)
                #             print('  WER (clean): %f %%' %
                #                   (wer_dev_clean_epoch * 100))

                #             metric_epoch = wer_dev_clean_epoch

                #             if metric_epoch < ler_dev_best:
                #                 ler_dev_best = metric_epoch
                #                 not_improved_epoch = 0
                #                 print('■■■ ↑Best Score (WER)↑ ■■■')

                #                 # Save model (check point)
                #                 checkpoint_file = join(
                #                     model.save_path, 'model.ckpt')
                #                 save_path = saver.save(
                #                     sess, checkpoint_file, global_step=train_data.epoch)
                #                 print("Model saved in file: %s" % save_path)

                #                 print('=== Test Data Evaluation ===')
                #                 # test-clean
                #                 # cer_test_clean_epoch = do_eval_wer(
                #                 #     session=sess,
                #                 #     decode_ops=decode_ops,
                #                 #     model=model,
                #                 #     dataset=test_clean_data,
                #                 #     train_data_size=params['train_data_size'],
                #                 #     is_test=True,
                #                 #     eval_batch_size=1)
                #                 # print('  WER (clean): %f %%' %
                #                 #       (cer_test_clean_epoch * 100))
                #                 #
                #                 # # test-other
                #                 # ler_test_other_epoch = do_eval_wer(
                #                 #     session=sess,
                #                 #     decode_ops=decode_ops,
                #                 #     model=model,
                #                 #     dataset=test_other_data,
                #                 #     train_data_size=params['train_data_size'],
                #                 #     is_test=True,
                #                 #     eval_batch_size=1)
                #                 # print('  WER (other): %f %%' %
                #                 #       (ler_test_other_epoch * 100))
                #             else:
                #                 not_improved_epoch += 1

                #         duration_eval = time.time() - start_time_eval
                #         print('Evaluation time: %.3f min' %
                #               (duration_eval / 60))

                #         # Early stopping
                #         if not_improved_epoch == params['not_improved_patient_epoch']:
                #             break

                #         # Update learning rate
                #         learning_rate = lr_controller.decay_lr(
                #             learning_rate=learning_rate,
                #             epoch=train_data.epoch,
                #             value=metric_epoch)

                #     start_time_step = time.time()
                #     start_time_epoch = time.time()

            duration_train = time.time() - start_time_train
            print('Total time: %.3f hour' % (duration_train / 3600))

            # Training was finished correctly
            with open(join(model.save_path, 'complete.txt'), 'w') as f:
                f.write('')


def main(config_path, model_save_path, gpu_indices):

    # Load a config file (.yml)
    with open(config_path, "r") as f:
        config = yaml.load(f)
        params = config['param']

    # Except for a blank class
    # TODO load vocab.txt num
    if params['label_type'] == 'character':
        params['num_classes'] = 4714
    else:
        raise TypeError

    # Model setting
    model = CTC(encoder_type=params['encoder_type'],
                input_size=params['input_size'],
                splice=params['splice'],
                num_stack=params['num_stack'],
                num_units=params['num_units'],
                num_layers=params['num_layers'],
                num_classes=params['num_classes'],
                lstm_impl=params['lstm_impl'],
                use_peephole=params['use_peephole'],
                parameter_init=params['weight_init'],
                clip_grad_norm=params['clip_grad_norm'],
                clip_activation=params['clip_activation'],
                num_proj=params['num_proj'],
                weight_decay=params['weight_decay'])

    # Set process name
    setproctitle(
        'tf_' + model.name + '_' + params['train_data_size'] + '_' + params['label_type'])

    model.name += '_' + str(params['num_units'])
    model.name += '_' + str(params['num_layers'])
    model.name += '_' + params['optimizer']
    model.name += '_lr' + str(params['learning_rate'])
    if params['num_proj'] != 0:
        model.name += '_proj' + str(params['num_proj'])
    if params['dropout'] != 0:
        model.name += '_drop' + str(params['dropout'])
    if params['num_stack'] != 1:
        model.name += '_stack' + str(params['num_stack'])
    if params['weight_decay'] != 0:
        model.name += '_wd' + str(params['weight_decay'])
    if params['bottleneck_dim'] != 0:
        model.name += '_bottle' + str(params['bottleneck_dim'])
    if len(gpu_indices) >= 2:
        model.name += '_gpu' + str(len(gpu_indices))

    # Set save path
    model.save_path = mkdir_join(
        model_save_path, 'ctc', params['label_type'],
        params['train_data_size'], model.name)

    # Reset model directory
    model_index = 0
    new_model_path = model.save_path
    while True:
        if isfile(join(new_model_path, 'complete.txt')):
            # Training of the first model have been finished
            model_index += 1
            new_model_path = model.save_path + '_' + str(model_index)
        elif isfile(join(new_model_path, 'config.yml')):
            # Training of the first model have not been finished yet
            model_index += 1
            new_model_path = model.save_path + '_' + str(model_index)
        else:
            break
    model.save_path = mkdir(new_model_path)

    # Save config file
    shutil.copyfile(config_path, join(model.save_path, 'config.yml'))

    #sys.stdout = open(join(model.save_path, 'train.log'), 'w')
    # TODO(hirofumi): change to logger
    do_train(model=model, params=params, gpu_indices=gpu_indices)


if __name__ == '__main__':
    # import pdb;pdb.set_trace()
    args = sys.argv
    if len(args) != 3 and len(args) != 4:
        raise ValueError
    main(config_path=args[1], model_save_path=args[2],
         gpu_indices=list(map(int, args[3].split(','))))



# if __name__ == "__main__":
#     files, durations, labels = [], [], []
#     import json
#     for j in open("/Users/fanlu/Downloads/aishell_label_dict.txt").readlines():
#         label_dict = {v: k for k, v in enumerate(j.strip().split())}
#     with open("/Users/fanlu/Downloads/aishell_train.json", 'r') as f:
#         for i, line in enumerate(f.readlines()):
#             d = json.loads(line)
#             files.append(d.get("key"))
#             durations.append(float(d.get("duration")))
#             labels.append(" ".join([str(label_dict.get(i)) for i in d.get("text").strip().split()]))
    
#     src_dataset = tf.data.Dataset.from_tensor_slices((files, durations, labels))
#     batch_size = 4
#     output_buffer_size = batch_size * 10
#     num_parallel_calls = 4

#     batched_iter = get_iterator2(src_dataset, None, None, None, batch_size, None, None, random_seed=3,
#                                  num_buckets=16, output_buffer_size=output_buffer_size,
#                                  num_parallel_calls=num_parallel_calls,
#                                  num_shards=2, shard_index=0, label_dict=label_dict)

#     batched_iter2 = get_iterator2(src_dataset, None, None, None, batch_size, None, None, random_seed=3,
#                                   num_buckets=16, output_buffer_size=output_buffer_size,
#                                   num_parallel_calls=num_parallel_calls,
#                                   num_shards=2, shard_index=1, label_dict=label_dict)
#     src_ids = batched_iter.source
#     src_seq_len = batched_iter.source_sequence_length
#     dur = batched_iter.duration
#     la = batched_iter.label
#     fi = batched_iter.filename

#     src_ids2 = batched_iter2.source
#     src_seq_len2 = batched_iter2.source_sequence_length
#     dur2 = batched_iter2.duration
#     la2 = batched_iter2.label
#     fi2 = batched_iter2.filename
#     import pdb;pdb.set_trace()
    
#     label_sparse = dense_to_sparse(la)
#     # table_initializer = tf.tables_initializer()
#     with tf.Session() as sess:
#         # sess.run(table_initializer)
#         sess.run(batched_iter.initializer)
#         sess.run(batched_iter2.initializer)
        
#         for i in range(100):
#             try:
#                 (source_v, src_len_v, duration, label, filename, source_v_2, src_len_v_2, duration2, label2,
#                  filename2) = (
#                     sess.run((src_ids, src_seq_len, dur, la, fi, src_ids2, src_seq_len2, dur2, la2, fi2)))
#                 l_s_v = sess.run(label_sparse)
#                 print(source_v.shape, src_len_v, source_v_2.shape, src_len_v_2)
#             #       for j in range(4):
#             #         print(filename[j].rsplit("/",1)[1], duration[j], filename2[j].rsplit("/",1)[1], duration2[j])
#             except tf.errors.OutOfRangeError:
#                 print('end', i)
#                 break
