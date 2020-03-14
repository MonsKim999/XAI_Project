
import pickle
import h5py
import cv2
import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity

Anns = pickle.load(open('/home/jungjunkim/xlmert_feat/anns.pkl', 'rb'))
# cxt_iid2id = pickle.load(open('/home/jungjunkim/xlmert_feat/train36_imgid2idx.pkl', 'rb'))
# cxt_features = h5py.File('/home/jungjunkim/xlmert_feat/train36.hdf5', 'r')

cxt_iid2id = pickle.load(open('/home/jungjunkim/generate_captions_for_vqa-master/data/train36_imgid2idx.pkl', 'rb'))
cxt_features = h5py.File('/home/jungjunkim/generate_captions_for_vqa-master/data/train36.hdf5', 'r')

all_features = cxt_features.get('image_features')
all_boxes = cxt_features.get('image_bb')
all_boxes2 = pickle.load(open('/home/jungjunkim/xlmert_feat/iid_to_boxes.pkl', 'rb'))

# iid_to_path = pickle.load(open('/home/jungjunkim/dataset/iid2path.pkl', 'rb'))
# target_feats = np.load('/home/jungjunkim/xlmert_feat/ann_feats.npy')
# target_feats = np.load('/home/jungjunkim/slr/save_dir/prepro/refcocog_google2/ann_feats.npy')
target_feats = np.load('/home/jungjunkim/Downloads/ann_feats.npy')
ix_to_cat = pickle.load(open('/home/jungjunkim/xlmert_feat/ix_to_cat.pkl', 'rb'))
cat_to_ix = {ix_to_cat[i]: i for i in ix_to_cat}


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

def drawing(image_path, box, n_th, image_id, question):

    image = cv2.imread(image_path)

    # draw the ground-truth bounding box along with the predicted
    # bounding box

    det_box = box
    det_box[0] = int(det_box[0])
    det_box[1] = int(det_box[1])
    det_box[2] = int(det_box[2]) + det_box[0]
    det_box[3] = int(det_box[3]) + det_box[1]

    # ref_id = entry['ref_id']
    # iid = entry['image_id'][0]
    # sentence = entry['sent']

    cv2.rectangle(image, tuple(det_box[:2]),
                  tuple(det_box[2:]), (0, 0, 255), 2)

    # cv2.putText(image, "{}".format(str(question[0]['question'])), (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    #
    # cv2.putText(image, "{}".format(str(question[1]['question'])), (10, 60),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite('/home/jungjunkim/study_on_bbox/{}_{}.jpg'.format(image_id, n_th), image)

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

def cal_cosim(tarfeat, neighbour_feats):

    cosim_list = []

    for jj in range(36):
        cosim = cosine_similarity(tarfeat.reshape(1, -1), neighbour_feats[jj].reshape(1, -1))
        cosim_list.append(cosim.item())

    return cosim_list

ii = 0

def draw_target_neighbour(image_path, target, neighbor, image_id, cosim, iou, eud, index, ann_id):

    image = cv2.imread(image_path)

    det_box = np.zeros_like(target)
    det_box[0] = int(target[0])
    det_box[1] = int(target[1])
    det_box[2] = int(target[2]) + det_box[0]
    det_box[3] = int(target[3]) + det_box[1]

    tbox = []
    for y in range(4):
        tbox.append(int(det_box[y]))

    cv2.rectangle(image, tuple(tbox[:2]),
                  tuple(tbox[2:]), (0, 0, 255), 2)

    neighbor_box = np.zeros_like(neighbor)
    neighbor_box[0] = int(neighbor[0])
    neighbor_box[1] = int(neighbor[1])
    neighbor_box[2] = int(neighbor[2]) + neighbor_box[0]
    neighbor_box[3] = int(neighbor[3]) + neighbor_box[1]

    nbox = []
    for yy in range(4):
        nbox.append(int(neighbor_box[yy]))

    cv2.rectangle(image, tuple(nbox[:2]),
                  tuple(nbox[2:]), (0, 255, 0), 2)

    cv2.putText(image, "{}".format('cosim:'+str(cosim)), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.putText(image, "{}".format('iou:'+str(iou)), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.putText(image, "{}".format('euclidean:' + str(eud)), (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imwrite('/home/jungjunkim/study_on_bbox/{}_{}_{}.jpg'.format(ann_id, image_id, index), image)

    det_box = 0
    neighbor_box = 0

for ann_id in Anns:
    entry = Anns[ann_id]
    ii += 1
    print("[{}/{}]".format(ii, len(Anns)))

    # get file name
    img_id = entry['image_id']
    len_iid = len(str(img_id))
    len_rest = 12 - len_iid
    id = ''

    for i in range(len_rest):
        id += '0'

    id += str(img_id)
    filename = 'COCO_train2014_' + id +'.jpg'
    file_path = '/home/jungjunkim/py-R-FCN-multiGPU/data/train2014/' + filename

    # get target box and boxes
    target_box = entry['box']

    # get target feature and features
    h5_id = entry['h5_id']
    idx = cxt_iid2id[img_id]
    target_feat = target_feats[h5_id]
    feats = all_features[idx]
    boxes = all_boxes[idx]

    for k in range(36):
        boundingbox = boxes[k]
        boundingbox[2] -= boundingbox[0]
        boundingbox[3] -= boundingbox[1]
        boxes[k] = boundingbox

    # calculate the iou between target and neighbour
    iou_list = cal_iou(target_box, boxes)

    # calculate the euclidean distance between and neighbour
    euclidean_dis_list = cal_neighbours(target_box, boxes)

    # calculate the cosine similarity between and neighbour
    cosim_list = cal_cosim(target_feat, feats)
    for ix in range(36):
        draw_target_neighbour(file_path, target_box, boxes[ix], img_id, cosim_list[ix], iou_list[ix], euclidean_dis_list[ix], ix, ann_id)

    # image_path, target, neighbors, image_id, cosim_list
    # entry['neighbour_ids'] = neighbour_ids


# pickle.dump(entries, open('VQA_referring_testdataset.pkl', 'wb'), protocol=2)
# print("finish")
# descending_list = sorted(range(len(euclidean_dis_list)), key=lambda k: euclidean_dis_list[k])
# top_5_list = descending_list[1:6] # except for itself
# neighbour_ids.append(top_5_list)