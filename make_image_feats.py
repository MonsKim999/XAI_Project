import pickle
import h5py
import numpy as np
Images = pickle.load(open('/home/jungjunkim/Images.pkl', 'rb'))
cxt_iid2id = pickle.load(open('/home/jungjunkim/xlmert_feat/train36_imgid2idx.pkl', 'rb'))
cxt_features = h5py.File('/home/jungjunkim/xlmert_feat/train36.hdf5', 'r')
all_features = cxt_features.get('image_features')

img_feat1 = np.load('/home/jungjunkim/slr/save_dir/prepro/refcocog_google/image_feats1.npy')
img_feat = np.load('/home/jungjunkim/slr/save_dir/prepro/refcocog_google/image_feats.npy')


placeholder = np.zeros_like(img_feat)

for iid in Images:

    entry = Images[iid]
    h5_id = entry['h5_id']
    image_id = entry['image_id']
    idx = cxt_iid2id[image_id]

    feat36 = all_features[idx]
    feat36_mean = feat36.mean(0).reshape(1, -1)

    print("h5_id: ", h5_id)
    placeholder[h5_id] = feat36_mean

np.save('/home/jungjunkim/image_feats.npy', placeholder)