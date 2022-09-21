#####
# This utility script finds duplciated entries in W_GT matrix to decide which frames were manually annotated.
# After finding the manual flags, this script writes them on a new pkl file.
# Author: Mosam Dabhi

import _pickle as cPickle
import numpy as np

cam_num = 2

with open("CAM_" + str(cam_num) + ".pkl", "rb") as fid:
    data = cPickle.load(fid)


##### Deduplication and finding G.T. indices
W = []
for idx in range(len(data)):
    W.append(data[idx]["W_GT"])
W = np.asarray(W)

confidence = []
continue_checking = True
counter_idx = 0

while continue_checking:
    tmp = W[counter_idx] - W
    flags = ~np.any(np.any(tmp, axis=1), axis=1)
    uniques = np.argwhere(flags == True)[:, 0]
    max_number_uniques = max(uniques)

    confidence.append(counter_idx)
    if uniques.shape[0] == 1:
        counter_idx = counter_idx + 1
    else:
        counter_idx = max_number_uniques + 1

    if max_number_uniques >= 700:
        continue_checking = False

W_ = np.zeros((W.shape[0], W.shape[1], 2))
W_[:] = np.nan
W_[confidence, :, :] = W[confidence, :, :]

data_ = []

for idx in range(len(data)):
    d = {"W_GT": W_[idx, :, :]}
    data_.append(d)


with open("CAM_" + str(cam_num) + ".pkl", "wb") as fid:
    cPickle.dump(data_, fid)
