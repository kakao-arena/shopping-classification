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
import cPickle
from itertools import izip

import fire
import h5py
import numpy as np


def evaluate(predict_path, data_path, div, y_vocab_path):
    h = h5py.File(data_path, 'r')[div]
    inv_y_vocab = {v: k
                   for k, v in cPickle.loads(open(y_vocab_path).read()).iteritems()}
    fin = open(predict_path)
    hit, n = {}, {'b': 0, 'm': 0, 's': 0, 'd': 0}
    print 'loading ground-truth...'
    CATE = np.argmax(h['cate'], axis=1)
    for p, y in izip(fin, CATE):
        pid, b, m, s, d = p.split('\t')
        b, m, s, d = map(int, [b, m, s, d])
        gt = map(int, inv_y_vocab[y].split('>'))
        for depth, _p, _g in zip(['b', 'm', 's', 'd'],
                                 [b, m, s, d],
                                 gt):
            if _g == -1:
                continue
            n[depth] = n.get(depth, 0) + 1
            if _p == _g:
                hit[depth] = hit.get(depth, 0) + 1
    for d in ['b', 'm', 's', 'd']:
        if n[d] > 0:
            print '%s-Accuracy: %.3f(%s/%s)' % (d, hit[d] / float(n[d]), hit[d], n[d])
    score = sum([hit[d] / float(n[d]) * w
                 for d, w in zip(['b', 'm', 's', 'd'],
                                 [1.0, 1.2, 1.3, 1.4])]) / 4.0
    print 'score: %.3f' % score

if __name__ == '__main__':
    fire.Fire({'evaluate': evaluate})
