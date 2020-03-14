import pickle

train_set = pickle.load(open('/home/jungjunkim/xlmert_feat/VQA_ref_traindataset.pkl', 'rb'))

new_entry = {}

for entry in train_set:
    image_id = entry['image_id']
    boxes = entry['boxes']

    new_entry .update({image_id: boxes})

pickle.dump(new_entry, open('/home/jungjunkim/xlmert_feat/iid_to_boxes.pkl', 'wb'))