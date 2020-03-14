import pickle

train_entries = pickle.load(open('/home/jungjunkim/xlmert_feat/VQA_ref_traindataset.pkl', 'rb'))
# val_entries = pickle.load(open('/home/jungjunkim/xlmert_feat/VQA_ref_valdataset.pkl', 'rb'))
# test_entries = pickle.load(open('/home/jungjunkim/xlmert_feat/VQA_ref_testdataset.pkl', 'rb'))


for entry in train_entries:
    print("k")