import torch
import numpy as np
from argparse import Namespace
from model import SASRec
from utils import data_partition

# --- config: set these to match training ---
dataset = 'Beauty'
state_dict_path = r'.\Beauty_default\SASRec.epoch=1000.lr=0.001.layer=2.head=1.hidden=50.maxlen=50.pth'
device = 'cuda'  # or 'cuda'
args = Namespace(
    device=device,
    hidden_units=50,
    maxlen=50,
    num_blocks=2,
    num_heads=1,
    dropout_rate=0.2,
    norm_first=False,
    l2_emb=0.0
)

# --- load dataset info to get item/user counts ---
dataset_meta = data_partition(dataset)
# data_partition returns [user_train, user_valid, user_test, usernum, itemnum]
user_train, user_valid, user_test, usernum, itemnum = dataset_meta

# --- build model and load weights ---
model = SASRec(usernum, itemnum, args).to(device)
state = torch.load(state_dict_path, map_location=device)
model.load_state_dict(state)
model.eval()

'''
# --- example 1: given a known user ---
# user id (scalar or batch)
user_id = 123             # example user id
# a full padded history with length == args.maxlen (np.int32)
# zeros are padding (index 0)
seq = np.zeros((1, args.maxlen), dtype=np.int32)
# fill seq with this user's recent history (right-aligned)
hist = user_train[user_id]  # list of item ids (earliest -> latest)
tail = hist[-(args.maxlen):] if len(hist) > 0 else []
seq[0, -len(tail):] = tail
'''

# --- example 2: given a randomized item sequence ---
# user id is not used in this case, can be any valid id
user_id = 0  # dummy user id
seq = np.zeros((1, args.maxlen), dtype=np.int32)

# Create a random sequence of items for testing
# random_seq_length = np.random.randint(5, args.maxlen + 1)
random_seq_length = 15
hist = np.random.choice(a=range(1, itemnum), size=random_seq_length, replace=True).tolist()
print ("Random history:\n", hist)
# fill seq with the random history (right-aligned)
tail = hist[-(args.maxlen):] if len(hist) > 0 else [] # if the history is longer than maxlen, take the last maxlen items (tail)
seq[0, -len(tail):] = tail


# candidate item list you want scores for (1 x num_candidates)
candidate_items = np.array([list(range(1, itemnum))], dtype=np.int32)  # example first 100 items -> range(1,101)
# call predict (model expects numpy arrays)
with torch.no_grad():
    scores = model.predict(np.array([user_id]), seq, candidate_items)  # shape (1, num_candidates)
scores = scores[0]  # numpy or tensor on device
k = 10  # top-k items to show
print(f'\nTop {k} candidates (item_id, score):')
topk = torch.topk(scores, k=k)
for idx, score in zip(topk.indices.cpu().numpy(), topk.values.cpu().numpy()):
    print(candidate_items[0, idx], score)