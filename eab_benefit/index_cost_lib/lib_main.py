import torch
import numpy as np

import ast
from csv import reader

from learned_cost.index_cost_lib.lib_model import make_model, self_attn_model, eval_on_test_set

# Setting random seed to facilitate reproduction
seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

# cuda environment is recommended
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(device)

data_set_file_name = './data/TPC_DS_10_by_query.csv'
with open(data_set_file_name, 'r') as ro:
    csv_reader = reader(ro)
    raw_data = list(csv_reader)
# [𝑂𝑡, log(𝑐𝑎𝑟𝑑), log(𝑟𝑜𝑤𝑠), 𝑑𝑖𝑠𝑡_𝑓𝑟𝑎𝑐, 𝑁𝑈𝐿𝐿_𝑓𝑟𝑎𝑐, 𝐼𝑡, 𝐼𝑜]
data_label = []
for i in range(0, len(raw_data[0])):
    lists = ast.literal_eval(raw_data[0][i])
    data_label.append(lists)

# Remove empty points
data = []
label = []
for i in range(0, len(data_label)):
    if len(data_label[i][0]) == 0:
        continue
    else:
        data.append(data_label[i][0])
        label.append(data_label[i][1])

# Find the maximum number of index optimizable operations
max_l = 0
min_l = 999
for i in range(0, len(data)):
    max_l = max(max_l, len(data[i]))  # max(max_l, max([len(d) for d in data]))
    min_l = min(min_l, len(data[i]))  # min(min_l, min([len(d) for d in data]))

# Pad data to facilitate batch training/testing
pad_data = []
mask = []
pad_element = [0 for i in range(0, 12)]
for data_point in data:
    new_data = []
    point_mask = [0 for i in range(0, max_l)]

    for j in range(0, len(data_point)):
        new_data.append(data_point[j][:-1])  # ?[:-1]
        point_mask[j] = 1

    if max_l - len(data_point) > 0:
        for k in range(0, max_l - len(data_point)):
            new_data.append(pad_element)

    pad_data.append(new_data)
    mask.append(point_mask)

pad_data = torch.tensor(pad_data)
mask = torch.tensor(mask)
label = torch.tensor(label)

test_data = pad_data
test_mask = mask
test_label = label

len_test = len(test_data)
print(f'Size of test dataset: {len_test}')  # 5335

dim1 = 32  # embedding size
dim2 = 64  # hidden dimension for prediction layer
dim3 = 128  # hidden dimension for FNN
n_encoder_layers = 6  # number of layer of attention encoder
n_heads = 8  # number of heads in attention
dropout_r = 0.2  # dropout ratio
bs = 20  # batch size

encoder_model, pooling_model = make_model(dim1, n_encoder_layers, dim3, n_heads, dropout=dropout_r)
model = self_attn_model(encoder_model, pooling_model, 12, dim1, dim2)
para_dict_loc = './model/LIB_query.pth'
model.load_state_dict(torch.load(para_dict_loc))
print(model)
print('')
model = model.to(device)

eval_on_test_set(model, bs, device, len_test,
                 test_data, test_label, test_mask)
