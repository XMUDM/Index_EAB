import torch

from model.model import QueryFormer
from model.trainer import eval_workload
from model.util import Normalizer
from model.util import seed_everything
from model.database_util import get_hist_file, get_job_table_sample

data_path = './data/imdb/'
hist_file = get_hist_file(data_path + 'histogram_string.csv')
cost_norm = Normalizer(-3.61192, 12.290855)


class Args:
    pass


encoding_ckpt = torch.load('checkpoints/encoding.pt')
encoding = encoding_ckpt['encoding']
checkpoint = torch.load('checkpoints/cost_model.pt', map_location='cpu')

seed_everything()
args = checkpoint['args']

model = QueryFormer(emb_size=args.embed_size, ffn_dim=args.ffn_dim, head_size=args.head_size,
                    dropout=args.dropout, n_layers=args.n_layers,
                    use_sample=True, use_hist=True,
                    pred_hid=args.pred_hid)
model.load_state_dict(checkpoint['model'])

device = 'cuda:0'
_ = model.to(device).eval()

to_predict = 'cost'
methods = {
    'get_sample': get_job_table_sample,
    'encoding': encoding,
    'cost_norm': cost_norm,
    'hist_file': hist_file,
    'model': model,
    'device': device,
    'bs': 512,
}

eval_workload('job-light', methods)
eval_workload('synthetic', methods)
