#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Load dataset for the CTC model (Librispeech corpus).
   In addition, frame stacking and skipping are used.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

from os.path import join, isfile
from tqdm import tqdm
import pickle
import numpy as np
import json
import concurrent.futures
from multiprocessing import cpu_count
from utils.dataset.ctc import DatasetBase
from utils.util import mkdir_join
from utils.util import read_manifest
from utils.inputs.htk import read, write
from utils.inputs.wav2feature_python_speech_features import wav2feature as w2f_psf
from utils.inputs.wav2feature_librosa import wav2feature as w2f_librosa


class Dataset(DatasetBase):

    def __init__(self, data_type, train_data_size, label_type, batch_size,
                 max_epoch=None, splice=1,
                 num_stack=1, num_skip=1,
                 shuffle=False, sort_utt=False, sort_stop_epoch=None,
                 progressbar=False, num_gpu=1,
                 dataset_root="", json_file_path="", params=""):
        """A class for loading dataset.
        Args:
            data_type (stirng): train or dev_clean or dev_other or
                test_clean or test_other
            train_data_size (string): train100h or train460h or train960h or aishell
                or kefu different hours
            label_type (stirng): character or character_capital_divide or word
            batch_size (int): the size of mini-batch
            max_epoch (int, optional): the max epoch. None means infinite loop.
            splice (int, optional): frames to splice. Default is 1 frame.
            num_stack (int, optional): the number of frames to stack
            num_skip (int, optional): the number of frames to skip
            shuffle (bool, optional): if True, shuffle utterances. This is
                disabled when sort_utt is True.
            sort_utt (bool, optional): if True, sort all utterances by the
                number of frames and utteraces in each mini-batch are shuffled.
                Otherwise, shuffle utteraces.
            sort_stop_epoch (int, optional): After sort_stop_epoch, training
                will revert back to a random order
            progressbar (bool, optional): if True, visualize progressbar
            num_gpu (int, optional): if more than 1, divide batch_size by num_gpu
            dataset_root: the path where wav stored
            json_file_path: json file path
        """
        super(Dataset, self).__init__()

        self.data_type = data_type
        self.train_data_size = train_data_size
        self.label_type = label_type
        self.batch_size = batch_size * num_gpu
        self.max_epoch = max_epoch
        self.splice = splice
        self.num_stack = num_stack
        self.num_skip = num_skip
        self.shuffle = shuffle
        self.sort_utt = sort_utt
        self.sort_stop_epoch = sort_stop_epoch
        self.progressbar = progressbar
        self.num_gpu = num_gpu

        self.is_test = True if 'test' in data_type else False
        self.padded_value = -1 if not self.is_test else None
        self.input_paths = []
        self.target_labels = []
        self.durations = []
        self.label_dict = {}
        for i in open("/Users/fanlu/Downloads/aishell_train.json").readlines():
            dic = json.loads(i)
            self.input_paths.append(dic.get("key"))
            self.target_labels.append(dic.get("text"))
            self.durations.append(dic.get("duration"))
        import pdb;pdb.set_trace()
        for j in open("/Users/fanlu/Downloads/aishell_label_dict.txt").readlines():
            self.label_dict = {v: k for k, v in enumerate(j.strip().split())}
        
        self.rest = set(range(0, len(self.input_paths), 1))


    def load_or_cal_mean_std(self):
        return_dict = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            feat_dim = 3 * 41
            feat = np.zeros((1, feat_dim))
            feat_squared = np.zeros((1, feat_dim))
            count = 0
            future_to_f = {
                executor.submit(w2f_psf, f): f for f in
                self.input_paths}
            for future in concurrent.futures.as_completed(future_to_f):
                # for f, data in zip(audio_paths, executor.map(spectrogram_from_file, audio_paths, overwrite=overwrite, noise_percent=noise_percent)):
                f = future_to_f[future]
                try:
                    next_feat = future.result().reshape(-1, feat_dim)
                    next_feat_squared = np.square(next_feat)
                    feat_vertically_stacked = np.concatenate((feat, next_feat)).reshape(-1, feat_dim)
                    feat = np.sum(feat_vertically_stacked, axis=0, keepdims=True)
                    feat_squared_vertically_stacked = np.concatenate(
                        (feat_squared, next_feat_squared)).reshape(-1, feat_dim)
                    feat_squared = np.sum(feat_squared_vertically_stacked, axis=0, keepdims=True)
                    count += float(next_feat.shape[0])
                except Exception as exc:
                    print('%r generated an exception: %s' % (f, exc))
            return_dict[1] = {'feat': feat, 'feat_squared': feat_squared, 'count': count}

        feat = np.sum(np.vstack([item['feat'] for item in return_dict.values()]), axis=0)
        count = sum([item['count'] for item in return_dict.values()])
        feat_squared = np.sum(np.vstack([item['feat_squared'] for item in return_dict.values()]), axis=0)

        self.feats_mean = feat / float(count)
        self.feats_std = np.sqrt(feat_squared / float(count) - np.square(self.feats_mean))
        # np.savetxt(
        #     generate_file_path(self.save_dir, self.model_name, 'feats_mean'), self.feats_mean)
        # np.savetxt(
        #     generate_file_path(self.save_dir, self.model_name, 'feats_std'), self.feats_std)
        # log.info("End calculating mean and std from samples")

    def _compute_mean_std(self, manifest_path, featurize_func, num_samples):
        """Compute mean and std from randomly sampled instances."""
        manifest = read_manifest(manifest_path)
        sampled_manifest = self._rng.sample(manifest, num_samples)
        features = []
        for instance in sampled_manifest:
            features.append(featurize_func(instance["audio_filepath"]))
        features = np.vstack(features)
        self._mean = np.mean(features, axis=0).reshape([1, -1])
        self._std = np.std(features, axis=1).reshape([1, -1])

    def apply(self, features, eps=1e-14):
        """Normalize features to be of zero mean and unit stddev.

        :param features: Input features to be normalized.
        :type features: ndarray
        :param eps:  added to stddev to provide numerical stablibity.
        :type eps: float
        :return: Normalized features.
        :rtype: ndarray
        """
        return (features - self._mean) / (self._std + eps)

    def write_to_file(self, filepath):
        """Write the mean and stddev to the file.

        :param filepath: File to write mean and stddev.
        :type filepath: basestring
        """
        np.savez(filepath, mean=self._mean, std=self._std)

    def _read_mean_std_from_file(self, filepath):
        """Load mean and std from file."""
        npzfile = np.load(filepath)
        self._mean = npzfile["mean"]
        self._std = npzfile["std"]

