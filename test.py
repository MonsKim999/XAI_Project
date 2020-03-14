import pickle

all_boxes = pickle.load(open('/home/jungjunkim/xlmert_feat/iid_to_boxes.pkl', 'rb'))
entry = {}
for iid in all_boxes:
    boxes = all_boxes[iid]

    box36 = []

    for k in range(36):
        box = boxes[k]
        box[2] = box[2] - box[0]
        box[3] = box[3] - box[1]
        box36.append(box)

    entry.update({iid: box36})

pickle.dump(entry, open('/home/jungjunkim/xlmert_feat/iid_to_boxes.pkl', 'wb'))
print("k")