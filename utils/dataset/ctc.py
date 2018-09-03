#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for loading dataset for the CTC model.
   In this class, all data will be loaded at each step.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import basename
import random
import numpy as np

from utils.dataset.base import Base
from utils.io.inputs.frame_stacking import stack_frame
from utils.io.inputs.splicing import do_splice
import soundfile
from python_speech_features import fbank, delta

def fbank_from_file(wav_path):
    sig1, sr1 = soundfile.read(wav_path, dtype='float32')
    fbank_feat, energy = fbank(sig1, sr1, nfilt=40)  # (407, 40)
    # fbank_feat = np.column_stack(
    #     (np.log(energy), np.log(fbank_feat)))  # (407, 41)
    d_fbank_feat = delta(fbank_feat, 2)
    dd_fbank_feat = delta(d_fbank_feat, 2)
    # concat_fbank_feat = np.array(
    #     [fbank_feat, d_fbank_feat, dd_fbank_feat], dtype=np.float32)  # (3, 407, 41)
    # return concat_fbank_feat
    return np.column_stack((np.log(fbank_feat), d_fbank_feat, dd_fbank_feat))


class DatasetBase(Base):

    def __init__(self, *args, **kwargs):
        super(DatasetBase, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        input_i = np.array(self.input_paths[index])
        label_i = np.array(self.label_paths[index])
        return (input_i, label_i)

    def text_2_indices(self, text):
        return [self.label_dict.get(i) for i in text.strip().split()]


    def __next__(self, batch_size=None):
        """Generate each mini-batch.
        Args:
            batch_size (int, optional): the size of mini-batch
        Returns:
            A tuple of `(inputs, labels, inputs_seq_len, input_names)`
                inputs: list of input data of size
                    `[num_gpu, B, T_in, input_size]`
                labels: list of target labels of size
                    `[num_gpu, B, T_out]`
                inputs_seq_len: list of length of inputs of size
                    `[num_gpu, B]`
                input_names: list of file name of input data of size
                    `[num_gpu, B]`
            is_new_epoch (bool): If true, 1 epoch is finished
        """
        import pdb;pdb.set_trace()
        if self.max_epoch is not None and self.epoch >= self.max_epoch:
            raise StopIteration
        # NOTE: max_epoch = None means infinite loop

        if batch_size is None:
            batch_size = self.batch_size

        # reset
        if self.is_new_epoch:
            self.is_new_epoch = False

        if not self.is_test:
            self.padded_value = -1
        else:
            self.padded_value = None
        # TODO(hirofumi): move this

        if self.sort_utt:
            # Sort all uttrances by length
            if len(self.rest) > batch_size:
                data_indices = sorted(list(self.rest))[:batch_size]
                self.rest -= set(data_indices)
                # NOTE: rest is uttrance length order
            else:
                # Last mini-batch
                data_indices = list(self.rest)
                self.reset()
                self.is_new_epoch = True
                self.epoch += 1
                if self.epoch == self.sort_stop_epoch:
                    self.sort_utt = False
                    self.shuffle = True

            # Shuffle data in the mini-batch
            random.shuffle(data_indices)

        elif self.shuffle:
            # Randomly sample uttrances
            if len(self.rest) > batch_size:
                data_indices = random.sample(list(self.rest), batch_size)
                self.rest -= set(data_indices)
            else:
                # Last mini-batch
                data_indices = list(self.rest)
                self.reset()
                self.is_new_epoch = True
                self.epoch += 1

                # Shuffle selected mini-batch
                random.shuffle(data_indices)

        else:
            if len(self.rest) > batch_size:
                data_indices = sorted(list(self.rest))[:batch_size]
                self.rest -= set(data_indices)
                # NOTE: rest is in name order
            else:
                # Last mini-batch
                data_indices = list(self.rest)
                self.reset()
                self.is_new_epoch = True
                self.epoch += 1
        
        # Load dataset in mini-batch
        # input_list = np.array(list(
        #     map(lambda path: np.load(path),
        #         np.take(self.input_paths, data_indices, axis=0))))
        input_list = np.array(list(
            map(lambda path: fbank_from_file(path),
                np.take(self.input_paths, data_indices, axis=0))))
        label_list = np.array(list(
            map(lambda text: self.text_2_indices(text),
                np.take(self.target_labels, data_indices, axis=0))))
        # import pdb
        # pdb.set_trace()
        # label_list = np.array(list([self.label_dict.get(i) for i in np.take(self.target_labels, data_indices, axis=0)]))

        if not hasattr(self, 'input_size'):
            self.input_size = input_list[0].shape[1]
            if self.num_stack is not None and self.num_skip is not None:
                self.input_size *= self.num_stack

        # Frame stacking
        input_list = stack_frame(input_list,
                                 self.num_stack,
                                 self.num_skip,
                                 progressbar=False)

        # Compute max frame num in mini-batch
        max_frame_num = max(map(lambda x: x.shape[0], input_list))

        # Compute max target label length in mini-batch
        max_seq_len = max(map(len, label_list))

        # Initialization
        inputs = np.zeros(
            (len(data_indices), max_frame_num, self.input_size * self.splice),
            dtype=np.float32)
        labels = np.array(
            [[self.padded_value] * max_seq_len] * len(data_indices))
        inputs_seq_len = np.zeros((len(data_indices),), dtype=np.int32)
        input_names = list(
            map(lambda path: basename(path).split('.')[0],
                np.take(self.input_paths, data_indices, axis=0)))
        import pdb;pdb.set_trace()
        # Set values of each data in mini-batch
        for i_batch in range(len(data_indices)):
            data_i = input_list[i_batch]
            frame_num, input_size = data_i.shape

            # Splicing
            data_i = data_i.reshape(1, frame_num, input_size)
            # data_i = do_splice(data_i,
            #                    splice=self.splice,
            #                    batch_size=1,
            #                    num_stack=self.num_stack)
            data_i = data_i.reshape(frame_num, -1)

            inputs[i_batch, :frame_num, :] = data_i
            if self.is_test:
                labels[i_batch, 0] = label_list[i_batch]
            else:
                labels[i_batch, :len(label_list[i_batch])
                       ] = label_list[i_batch]
            inputs_seq_len[i_batch] = frame_num

        ###############
        # Multi-GPUs
        ###############
        if self.num_gpu > 1:
            # Now we split the mini-batch data by num_gpu
            inputs = np.array_split(inputs, self.num_gpu, axis=0)
            labels = np.array_split(labels, self.num_gpu, axis=0)
            inputs_seq_len = np.array_split(
                inputs_seq_len, self.num_gpu, axis=0)
            input_names = np.array_split(input_names, self.num_gpu, axis=0)
        else:
            inputs = inputs[np.newaxis, :, :, :]
            labels = labels[np.newaxis, :, :]
            inputs_seq_len = inputs_seq_len[np.newaxis, :]
            input_names = np.array(input_names)[np.newaxis, :]

        self.iteration += len(data_indices)

        # Clean up
        del input_list
        del label_list

        return (inputs, labels, inputs_seq_len, input_names), self.is_new_epoch
