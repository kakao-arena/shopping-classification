# -*- coding: utf-8 -*-
# Copyright 2017 Kakao, Recommendation Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import cPickle
from itertools import izip

import fire
import h5py
import numpy as np

from keras.models import load_model
from keras.callbacks import ModelCheckpoint

from misc import get_logger, Option
from network import TextOnly, top1_acc

opt = Option('./config.json')
cate1 = json.loads(open('../cate1.json').read())
DEV_DATA_LIST = ['../dev.chunk.01']


class Classifier():
    def __init__(self):
        self.logger = get_logger('Classifier')
        self.num_classes = 0

    def get_sample_generator(self, ds, batch_size):
        left, limit = 0, ds['uni'].shape[0]
        while True:
            right = min(left + batch_size, limit)
            X = [ds[t][left:right, :] for t in ['uni', 'w_uni']]
            Y = ds['cate'][left:right]
            yield X, Y
            left = right
            if right == limit:
                left = 0

    def get_inverted_cate1(self, cate1):
        inv_cate1 = {}
        for d in ['b', 'm', 's', 'd']:
            inv_cate1[d] = {v: k for k, v in cate1[d].iteritems()}
        return inv_cate1

    def write_prediction_result(self, data, pred_y, meta, out_path, readable):
        pid_order = []
        for data_path in DEV_DATA_LIST:
            h = h5py.File(data_path, 'r')['dev']
            pid_order.extend(h['pid'][::])

        y2l = {i: s for s, i in meta['y_vocab'].iteritems()}
        y2l = map(lambda x: x[1], sorted(y2l.items(), key=lambda x: x[0]))
        inv_cate1 = self.get_inverted_cate1(cate1)
        rets = {}
        for pid, p in izip(data['pid'], pred_y):
            y = np.argmax(p)
            label = y2l[y]
            tkns = map(int, label.split('>'))
            b, m, s, d = tkns
            assert b in inv_cate1['b']
            assert m in inv_cate1['m']
            assert s in inv_cate1['s']
            assert d in inv_cate1['d']
            tpl = '{pid}\t{b}\t{m}\t{s}\t{d}'
            if readable:
                b = inv_cate1['b'][b]
                m = inv_cate1['m'][m]
                s = inv_cate1['s'][s]
                d = inv_cate1['d'][d]
            rets[pid] = tpl.format(pid=pid, b=b, m=m, s=s, d=d)
        no_answer = '{pid}\t-1\t-1\t-1\t-1'
        with open(out_path, 'w') as fout:
            for pid in pid_order:
                ans = rets.get(pid, no_answer.format(pid=pid))
                print >> fout, ans

    def predict(self, data_root, model_root, test_root, test_div, out_path, readable=False):
        meta_path = os.path.join(data_root, 'meta')
        meta = cPickle.loads(open(meta_path).read())

        model_fname = os.path.join(model_root, 'model.h5')
        self.logger.info('# of classes(train): %s' % len(meta['y_vocab']))
        model = load_model(model_fname,
                           custom_objects={'top1_acc': top1_acc})

        test_path = os.path.join(test_root, 'data.h5py')
        test_data = h5py.File(test_path, 'r')

        test = test_data[test_div]
        test_gen = self.get_sample_generator(test, opt.batch_size)
        total_test_samples = test['uni'].shape[0]
        steps = int(np.ceil(total_test_samples / float(opt.batch_size)))
        pred_y = model.predict_generator(test_gen,
                                         steps=steps,
                                         workers=opt.num_predict_workers,
                                         verbose=1)
        self.write_prediction_result(test, pred_y, meta, out_path, readable=readable)

    def train(self, data_root, out_dir):
        data_path = os.path.join(data_root, 'data.h5py')
        meta_path = os.path.join(data_root, 'meta')
        data = h5py.File(data_path, 'r')
        meta = cPickle.loads(open(meta_path).read())
        self.weight_fname = os.path.join(out_dir, 'weights')
        self.model_fname = os.path.join(out_dir, 'model')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        self.logger.info('# of classes: %s' % len(meta['y_vocab']))
        self.num_classes = len(meta['y_vocab'])

        train = data['train']
        dev = data['dev']

        self.logger.info('# of train samples: %s' % train['cate'].shape[0])
        self.logger.info('# of dev samples: %s' % dev['cate'].shape[0])

        checkpoint = ModelCheckpoint(self.weight_fname, monitor='val_loss',
                                     save_best_only=True, mode='min', period=10)

        textonly = TextOnly()
        model = textonly.get_model(self.num_classes)

        total_train_samples = train['uni'].shape[0]
        train_gen = self.get_sample_generator(train,
                                              batch_size=opt.batch_size)
        self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt.batch_size)))

        total_dev_samples = dev['uni'].shape[0]
        dev_gen = self.get_sample_generator(dev,
                                            batch_size=opt.batch_size)
        self.validation_steps = int(np.ceil(total_dev_samples / float(opt.batch_size)))

        model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=[checkpoint])

        model.load_weights(self.weight_fname) # loads from checkout point if exists
        open(self.model_fname + '.json', 'w').write(model.to_json())
        model.save(self.model_fname + '.h5')


if __name__ == '__main__':
    clsf = Classifier()
    fire.Fire({'train': clsf.train,
               'predict': clsf.predict})
