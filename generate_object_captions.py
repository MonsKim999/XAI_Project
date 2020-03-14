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
from scipy.spatial import distance

train_entries = pickle.load(open('/home/hjung/dataset/VQA_ref_testdataset.pkl', 'rb'))
# val_entries = pickle.load(open('/home/jungjunkim/xlmert_feat/VQA_ref_valdataset.pkl', 'rb'))
# test_entries = pickle.load(open('/home/jungjunkim/xlmert_feat/VQA_ref_testdataset.pkl', 'rb'))

train_iid2id = pickle.load(open('/home/hjung/dataset/test36_imgid2idx.pkl', 'rb'))
# val_iid2id = pickle.load(open('/home/jungjunkim/generate_captions_for_vqa-master/data/val36_imgid2idx.pkl', 'rb'))
# test_iid2id = pickle.load(open('/home/jungjunkim/generate_captions_for_vqa-master/data/test36_imgid2idx.pkl', 'rb'))

cxt_features = h5py.File('/home/hjung/dataset/test36.hdf5', 'r')
all_image_feats = cxt_features.get('image_features')
all_boxes = cxt_features.get('image_bb')
sizes = pd.read_csv('test_image_size.csv')

def get_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def cal_iou(target_box, boxes):
    iouList = []

    tbox = np.zeros_like(target_box)
    abox = np.zeros_like(target_box)

    for ii in range(36):

        tbox[0] = target_box[0]
        tbox[1] = target_box[1]
        tbox[2] = tbox[0] + target_box[2]
        tbox[3] = tbox[1] + target_box[3]

        abox[0] = boxes[ii][0]
        abox[1] = boxes[ii][1]
        abox[2] = boxes[ii][2] + abox[0]
        abox[3] = boxes[ii][3] + abox[1]

        iou = get_iou(tbox, abox)
        iouList.append(iou)

    return iouList

def cal_centroid(bbox):
    return bbox[0] + (bbox[2] / 2), bbox[1] + (bbox[3] / 2)

def cal_euclidean(target, neighbour):
    cent1 = cal_centroid(target)
    cent2 = cal_centroid(neighbour)

    dis = distance.euclidean(cent1, cent2)

    return dis

def cal_neighbours(target_box, neighbour_boxes):
    dis_list = []

    for i in range(36):
        dis = cal_euclidean(target_box, neighbour_boxes[i])
        dis_list.append(dis)

    return dis_list

def fetch_my_neighbour(target_box, boxes36, image_id):

    idx = train_iid2id[image_id]

    features36 = all_image_feats[idx]

    iou_list = cal_iou(target_box, boxes36)
    iou_threshold = 0.2
    extracted_iou_inds = np.where(np.array(iou_list) < iou_threshold)

    euclidean_list = cal_neighbours(target_box, boxes36)
    extracted_euclidean_list = []

    for ind in list(extracted_iou_inds[0]):
        extracted_euclidean_list.append(euclidean_list[ind])

    descending_list = sorted(range(len(extracted_euclidean_list)), key=lambda k: extracted_euclidean_list[k])
    top_k = 20
    if len(descending_list) < top_k:
        top_k_list = descending_list
    else:
        top_k_list = descending_list[: top_k]  # except for itself

    cand_ann_feats = np.zeros((top_k, 2048), dtype=np.float32)
    neighbour_boxes = []
    for ix, top_k_ind in enumerate(top_k_list):
        cand_ann_feats[ix] = features36[top_k_ind]
        neighbour_boxes.append(list(boxes36[top_k_ind]))

    return cand_ann_feats, neighbour_boxes

def fetch_feats(entry, features, boxes, loader, ix, opt):

    expand_size = opt['seq_per_ref']
    batch_size = 1

    dif_ann_feats = np.zeros((batch_size, 2048), dtype=np.float32)
    dif_lfeats = np.zeros((batch_size, 5 * 20), dtype=np.float32)
    image_id = entry['image_id']

    for i in range(batch_size):

        rbox = boxes[ix]

        cand_ann_feats, neighbour_boxes = fetch_my_neighbour(rbox, boxes, image_id)

        ref_ann_feat = features[ix]
        dif_ann_feat = np.mean(cand_ann_feats - ref_ann_feat, axis=0)

        rcx, rcy, rw, rh = rbox[0] + rbox[2] / 2, rbox[1] + rbox[3] / 2, rbox[2], rbox[3]
        dif_lfeat = []

        for neighbor_box in neighbour_boxes:

            cbox = neighbor_box
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
        boxes = all_boxes[idx]
        referring_expression = []

        for m in range(36):
            bbox = boxes[m]
            bbox[2] -= bbox[0]
            bbox[3] -= bbox[1]
            boxes[m] = bbox

        for k in range(36):

            feats = fetch_feats(entry, features, boxes, loader, k, params)
            feats = Variable(xp.array(feats, dtype=xp.float32))

            vis_enc_feats = ve(feats)
            lang_enc_feats = vis_enc_feats

            _, vis_emb_feats = cca(vis_enc_feats, lang_enc_feats)
            vis_feats = vis_combine(vis_enc_feats, vis_emb_feats)

            beam_results = beam_search(lm, vis_feats, params['beam_width'])
            results = [result['sent'] for result in beam_results[0]]

            results = results[:3]
            gen_sentence = []
            for i, result in enumerate(results):
                gen_sentence.append(' '.join([loader.ix_to_word[str(w)] for w in result]))
            referring_expression.append(gen_sentence)
        entry['object_captions'] = referring_expression
    pickle.dump(train_entries, open('VQA_ref_testdataset_v3.pkl', 'wb'))

# python eval_generation.py -id 96 -beam 3
if __name__ == '__main__':
    args = config.parse_opt()
    params = vars(args)  # convert to ordinary dict
    eval_all(params)