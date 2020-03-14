import sys
sys.path.append("cider")
from pyciderevalcap.ciderD.ciderD import CiderD
from pyciderevalcap.cider.cider import Cider
sys.path.append("coco-caption")
# from pycocoevalcap.bleu.bleu import Bleu
from collections import OrderedDict
import numpy as np

CiderD_scorer = None
Cider_scorer = None

#CiderD_scorer = CiderD(df='corpus')

def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    global Cider_scorer
    Cider_scorer = Cider_scorer or Cider(df=cached_tokens)

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

def get_self_critical_reward(greedy_res, data_gts, gen_result):
    batch_size = len(data_gts)
    gen_result_size = len(gen_result)
    seq_per_img = gen_result_size // len(data_gts) # gen_result_size  = batch_size * seq_per_img
    # assert greedy_res.shape[0] == batch_size

    res = OrderedDict()
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    for i in range(gen_result_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[gen_result_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    res_ = [{'image_id': i, 'caption': res[i]} for i in range(len(res))]
    res__ = {i: res[i] for i in range(len(res_))}
    gts_ = {i: gts[i // seq_per_img] for i in range(gen_result_size)}
    gts_.update({i+gen_result_size: gts[i] for i in range(batch_size)})
    _, cider_scores = CiderD_scorer.compute_score(gts_, res_)
    print('Cider scores:', _)

    cider_reward_weight = 1

    scores = cider_reward_weight * cider_scores

    scores = scores[:gen_result_size].reshape(batch_size, seq_per_img) - scores[-batch_size:][:, np.newaxis]
    scores = scores.reshape(gen_result_size)

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards