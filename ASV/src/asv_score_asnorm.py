import pandas as pd
import yaml, pickle
import argparse, warnings
from sklearn.metrics.pairwise import cosine_similarity as cs_sklearn
from SpeakerNet import *

parser = argparse.ArgumentParser(description = "Asnorm")
parser.add_argument('--embd_1200',      type=str,   default="/kaggle/working/embd_1200.pk", help='Absolute path to the embeddings file')
parser.add_argument('--asv_score',      type=str,   default="/kaggle/working/asv_score.txt", help='Absolute path to the score file')

#Initialize
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"
params={
        "cohort_size": 1200
    }
with open(args.embd_1200, "rb") as f:
    embd_trn = pickle.load(f)
N = 300
selected_trials = list(embd_trn.keys())[:N * 2]
tensor_list = [torch.FloatTensor(arr) for arr in embd_trn.values()]

def asnorm_score(embd_trn, train_cohort):
    # Tính điểm số cho các cặp thử nghiệm và lưu vào một tensor
    cs = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    mean_e_list = []
    mean_t_list = []
    std_e_list = []
    std_t_list = []
    for i in range(0, len(selected_trials), 2):
        try:
            enrol = torch.FloatTensor(embd_trn[selected_trials[i+1]])
            test = torch.FloatTensor(embd_trn[selected_trials[i]])
        except:
            enrol = (embd_trn[selected_trials[i+1]])
            test = (embd_trn[selected_trials[i]])

        # Getting norm stats for enrol impostors
        enrol_rep = enrol.repeat(train_cohort.shape[0], 1, 1)
        score_e_c = cs(enrol_rep.to(device), train_cohort.to(device)).detach().cpu().numpy()

        if "cohort_size" in params:
            score_e_c = torch.topk(
                torch.Tensor(score_e_c), k=params["cohort_size"], dim=0
            )[0].detach().cpu().numpy()

        mean_e_c = torch.mean(torch.FloatTensor(score_e_c), dim=0)
        std_e_c = torch.std(torch.FloatTensor(score_e_c), dim=0)
        mean_e_list.append(mean_e_c)
        std_e_list.append(std_e_c)
        # Getting norm stats for test impostors
        test_rep = test.repeat(train_cohort.shape[0], 1, 1)
        score_t_c = cs(test_rep.to(device), train_cohort.to(device)).detach().cpu().numpy()

        if "cohort_size" in params:
            score_t_c = torch.topk(torch.Tensor(score_t_c), k=params["cohort_size"], dim=0)[0].detach().cpu().numpy()

        mean_t_c = torch.mean(torch.FloatTensor(score_t_c), dim=0)
        std_t_c = torch.std(torch.FloatTensor(score_t_c), dim=0)
        mean_t_list.append(mean_t_c)
        std_t_list.append(std_t_c)

    return mean_e_list, mean_t_list, std_e_list, std_t_list

def cal_norm_score(mean_e_list, mean_t_list, std_e_list, std_t_list, asv_score):
    norm_score = []
    scores = []
    asv_score_float = torch.tensor(asv_score, dtype=torch.float)

    for (me, mt, se, st) in zip(mean_e_list, mean_t_list, std_e_list, std_t_list):
        score_e = (asv_score_float - me) / se
        score_t = (asv_score_float - mt) / st
        score_e = torch.mean(score_e)
        score_t = torch.mean(score_t)
        
        score = 0.5 * (score_e + score_t)
        norm_score.append(score.item())
    scores.append(np.array(norm_score).mean())    
    return scores, norm_score

mean_e_list, mean_t_list, std_e_list, std_t_list = asnorm_score(embd_trn, torch.stack(tensor_list))
asv_scores = []
asv_scaled_scores = []
pairs = []
submission = []
with open(args.asv_score, "r") as f1:
    for line in f1:
        asv_scores.append(float(line.split('\t')[2]))
        pairs.append(line.split('\t')[:2])

for i in range(len(asv_scores)):
    asv_norm_score, norm_score = cal_norm_score(mean_e_list, mean_t_list, std_e_list, std_t_list, asv_scores[i])
    asv_scaled = (asv_norm_score - np.min(norm_score)) / (np.max(norm_score) - np.min(norm_score))
    # asv_scaled_scores.append(asv_scaled)
    submission.append([pairs[i][0], pairs[i][1], str(asv_scaled)])
    print(f'line:{i}\n')

with open("/kaggle/working/submission.txt", "w") as f2:
    for line in submission:
        f2.write('\t'.join(line) + '\n') 