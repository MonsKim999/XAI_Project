import argparse
import os.path as osp
import numpy as np
import json

import chainer
from chainer import Variable, cuda, serializers

from misc.DataLoader import DataLoader
from misc.utils import calc_max_ind, beam_search#, sample_function
from models.base import VisualEncoder, LanguageEncoder, LanguageEncoderAttn
from models.Listener import CcaEmbedding
from models.LanguageModel import vis_combine, LanguageModel
from misc.eval_utils import compute_margin_loss, computeLosses, language_eval
import config
import pickle
import h5py
import cupy
import os
import pandas as pd
import chainer.links as L
import chainer.functions as F

train_entries = pickle.load(open('/home/jungjunkim/xlmert_feat/VQA_ref_traindataset.pkl', 'rb'))
# val_entries = pickle.load(open('/home/jungjunkim/xlmert_feat/VQA_ref_valdataset.pkl', 'rb'))
# test_entries = pickle.load(open('/home/jungjunkim/xlmert_feat/VQA_ref_testdataset.pkl', 'rb'))

train_iid2id = pickle.load(open('/home/jungjunkim/xlmert_feat/train36_imgid2idx.pkl', 'rb'))
# val_iid2id = pickle.load(open('/home/jungjunkim/xlmert_feat/val36_imgid2idx.pkl', 'rb'))
# test_iid2id = pickle.load(open('/home/jungjunkim/xlmert_feat/test36_imgid2idx.pkl', 'rb'))

cxt_features = h5py.File('/home/jungjunkim/xlmert_feat/train36.hdf5', 'r')
all_image_feats = cxt_features.get('image_features')
all_boxes = cxt_features.get('image_bb')
sizes = pd.read_csv('trainval_image_size.csv')

def fetch_feats(entry, features, boxes, loader, ix, opt):

    expand_size = opt['seq_per_ref']
    batch_size = 1

    dif_ann_feats = np.zeros((batch_size, 2048), dtype=np.float32)
    dif_lfeats = np.zeros((batch_size, 5 * 5), dtype=np.float32)
    for i in range(batch_size):
        st_ann_ids = entry['same_category'][ix]
        st_ann_ids = st_ann_ids[:5]
        if len(st_ann_ids) == 0:
            continue

        cand_ann_feats = features[[st_id_ for st_id_ in st_ann_ids]]
        ref_ann_feat = features[ix]
        dif_ann_feat = np.mean(cand_ann_feats - ref_ann_feat, axis=0)
        rbox = boxes[ix]
        rcx, rcy, rw, rh = rbox[0] + rbox[2] / 2, rbox[1] + rbox[3] / 2, rbox[2], rbox[3]
        dif_lfeat = []
        for st_ann_id in st_ann_ids:
            cbox = boxes[st_ann_id]
            cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
            dif_lfeat.extend(
                [(cx1 - rcx) / rw, (cy1 - rcy) / rh, (cx1 + cw - rcx) / rw, (cy1 + ch - rcy) / rh, cw * ch / (rw * rh)])
        dif_ann_feats[i] = dif_ann_feat
        dif_lfeats[i, :len(dif_lfeat)] = dif_lfeat

    cxt_feats = features.mean(0).reshape(1, -1)  # (1, 2048)
    ann_feats = features[ix].reshape(1, -1) # (1, 2048)

    cxt_feats = np.tile(cxt_feats.reshape((batch_size, 1, -1)), (1, expand_size, 1)).reshape(
        (batch_size * expand_size, -1))
    ann_feats = np.tile(ann_feats.reshape((batch_size, 1, -1)), (1, expand_size, 1)).reshape(
        (batch_size * expand_size, -1))

    x, y, w, h = boxes[ix]
    image_info = sizes[str(entry['image_id'])]
    iw, ih = image_info[0], image_info[1]

    l_feats = np.array([x / iw, y / ih, (x + w - 1) / iw, (y + h - 1) / ih, w * h / (iw * ih)]).reshape(1, -1)
    l_feats = np.tile(np.array(l_feats, dtype=np.float32).reshape((batch_size, 1, -1)), (1, expand_size, 1)).reshape(
        (batch_size * expand_size, -1))


    dif_lfeat = np.array(dif_lfeats).reshape(1, -1)

    df_feats = np.tile(dif_ann_feats.reshape((batch_size, 1, -1)), (1, expand_size, 1)).reshape((batch_size * expand_size, -1))
    dlf_feats = np.tile(dif_lfeat.reshape((batch_size, 1, -1)), (1, expand_size, 1)).reshape((batch_size * expand_size, -1))

    feats = np.concatenate([cxt_feats, ann_feats, l_feats, df_feats, dlf_feats], axis=1)
    return feats


def eval_all(params):
    target_save_dir = osp.join(params['save_dir'], 'prepro', params['dataset'] + '_' + params['splitBy'])
    model_dir = osp.join(params['save_dir'], 'model', params['dataset'] + '_' + params['splitBy'])
    batch_size = params['batch_size']
    if params['old']:
        params['data_json'] = 'old' + params['data_json']
        params['data_h5'] = 'old' + params['data_h5']
        params['image_feats'] = 'old' + params['image_feats']
        params['ann_feats'] = 'old' + params['ann_feats']
        params['id'] = 'old' + params['id']

    loader = DataLoader(params)

    featsOpt = {'ann': osp.join(target_save_dir, params['ann_feats']),
                'img': osp.join(target_save_dir, params['image_feats'])}
    loader.loadFeats(featsOpt)
    chainer.config.train = False
    chainer.config.enable_backprop = False

    gpu_id = params['gpu_id']
    cuda.get_device(gpu_id).use()
    xp = cuda.cupy

    if 'attention' in params['id']:
        print('attn')
        le = LanguageEncoderAttn(len(loader.ix_to_word)).to_gpu(gpu_id)
    else:
        le = LanguageEncoder(len(loader.ix_to_word)).to_gpu(gpu_id)
    ve = VisualEncoder().to_gpu(gpu_id)
    cca = CcaEmbedding().to_gpu(gpu_id)
    lm = LanguageModel(len(loader.ix_to_word), loader.seq_length).to_gpu(gpu_id)

    serializers.load_hdf5(osp.join(model_dir, params['id'] + params['id2'] + "ve.h5"), ve)
    serializers.load_hdf5(osp.join(model_dir, params['id'] + params['id2'] + "le.h5"), le)
    serializers.load_hdf5(osp.join(model_dir, params['id'] + params['id2'] + "cca.h5"), cca)
    serializers.load_hdf5(osp.join(model_dir, params['id'] + params['id2'] + "lm.h5"), lm)

    for num, entry in enumerate(train_entries):
        print("{}/{}".format(num, len(train_entries)))

        image_id = entry['image_id']
        idx = train_iid2id[image_id]

        features = all_image_feats[idx]
        boxes = entry['boxes']
        referring_expression = []

        for k in range(36):

            feats = fetch_feats(entry, features, boxes, loader, k, params)
            feats = Variable(xp.array(feats, dtype=xp.float32))

            vis_enc_feats = ve(feats)
            lang_enc_feats = vis_enc_feats

            _, vis_emb_feats = cca(vis_enc_feats, lang_enc_feats)
            vis_feats = vis_combine(vis_enc_feats, vis_emb_feats)

            beam_results = beam_search(lm, vis_feats, params['beam_width'])
            results = [result['sent'] for result in beam_results[0]]
            # sample_results = lm.sample(vis_feats, stochastic=True)
            # results = [result for result in sample_results[0]]

            results = results[:3]
            gen_sentence = []
            for i, result in enumerate(results):
                gen_sentence.append(' '.join([loader.ix_to_word[str(w)] for w in result]))
            referring_expression.append(gen_sentence)
        entry['referring_expression'] = referring_expression
    # pickle.dump(train_entries, open('VQA_referring_testdataset_v2.pkl', 'wb'))

# python eval_generation.py -id 96 -beam 3
if __name__ == '__main__':
    args = config.parse_opt()
    params = vars(args)  # convert to ordinary dict
    eval_all(params)