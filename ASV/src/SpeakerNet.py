import torch
import torch.nn as nn
import torch.nn.functional as F
from DatasetLoader import test_dataset_loader
from torch.cuda.amp import autocast, GradScaler

import numpy
import sys
import random
import time
import itertools
import importlib
from scipy.spatial.distance import cdist
import numpy as np
import tqdm
import soundfile
import os
import pickle as pk

class WrappedModel(nn.Module):

    # The purpose of this wrapper is to make the model structure consistent between single and multi-GPU

    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None):
        return self.module(x, label)


class SpeakerNet(nn.Module):
    def __init__(self, model, trainfunc='aamsoftmax', nPerSpeaker=1, **kwargs):
        super(SpeakerNet, self).__init__()

        SpeakerNetModel = importlib.import_module(
            "models." + model).__getattribute__("MainModel")
        self.__S__ = SpeakerNetModel(**kwargs)

        LossFunction = importlib.import_module(
            "loss." + trainfunc).__getattribute__("LossFunction")
        self.__L__ = LossFunction(**kwargs)

        self.nPerSpeaker = nPerSpeaker

    def forward(self, data, label=None):

        data = data.reshape(-1, data.size()[-1]).cuda()
        # if isinstance(self.__S__, ECAPA_TDNN):
        #     outp = self.__S__.forward(data, aug=True)
        # else:
        outp = self.__S__.forward(data)

        if label == None:
            return outp
        else:
            outp = outp.reshape(self.nPerSpeaker, -1,
                                outp.size()[-1]).transpose(1, 0).squeeze(1)
            # clustering embeddings
            nloss, prec1 = self.__L__.forward(outp, label)

            return nloss, prec1


class ModelTrainer(object):
    def __init__(self, speaker_model, optimizer, scheduler, gpu, mixedprec, **kwargs):

        self.__model__ = speaker_model

        Optimizer = importlib.import_module(
            "optimizer." + optimizer).__getattribute__("Optimizer")
        self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)

        Scheduler = importlib.import_module(
            "scheduler." + scheduler).__getattribute__("Scheduler")
        self.__scheduler__, self.lr_step = Scheduler(
            self.__optimizer__, **kwargs)

        self.scaler = GradScaler()

        self.gpu = gpu

        self.mixedprec = mixedprec

        assert self.lr_step in ["epoch", "iteration"]

    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader, verbose):

        self.__model__.train()

        stepsize = loader.batch_size

        counter = 0
        index = 0
        loss = 0
        top1 = 0

        # EER or accuracy
        for data, data_label in loader:

            data = data.transpose(1, 0)

            self.__model__.zero_grad()

            label = torch.LongTensor(data_label).cuda()

            if self.mixedprec:
                with autocast():
                    nloss, prec1 = self.__model__(data, label)
                self.scaler.scale(nloss).backward()
                self.scaler.step(self.__optimizer__)
                self.scaler.update()
            else:
                nloss, prec1 = self.__model__(data, label)
                nloss.backward()
                self.__optimizer__.step()

            loss += nloss.detach().cpu().item() #accumulate loss over 'counter' number of batches
            top1 += prec1.detach().cpu().item() #accumulate accuracy over 'counter' number of batches
            counter += 1
            index += stepsize

            if verbose:
                sys.stderr.write(
                    "Training {:d} / {:d}:".format(index, loader.__len__() * loader.batch_size))
                sys.stderr.write(time.strftime("%m-%d %H:%M:%S") +
                                 "At current batch: Loss %.5f, Acc %.5f, LR %.7f \r" % (loss/counter, top1/counter, max([x['lr'] for x in self.__optimizer__.param_groups])))
                sys.stdout.flush()

            if self.lr_step == "iteration":
                self.__scheduler__.step()

        if self.lr_step == "epoch":
            self.__scheduler__.step()

        return (loss / counter, top1 / counter)

    # ===== ===== ===== ===== ===== ===== ===== =====
    # Evaluate from list
    # ===== ===== ===== ===== ===== ===== ===== =====

    def eval_network(self, test_list, test_path,eval_frames, **kwargs):
        # eval on valid_list.txt
        # [1 enr_1.wav test_1.wav]
        self.__model__.eval()
        files = []
        embeddings = {}
        spk_emb_dic = {}  # save embeddings of an utterance
        # lines = ["enr1.wav test1.wav"]
        lines = open(test_list).read().splitlines()
        for line in lines:
            files.append(line.split()[1])
            files.append(line.split()[2])
        setfiles = list(set(files))
        setfiles.sort()

        for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
            audio, _ = soundfile.read(os.path.join(test_path, file))
            # Full utterance
            data_1 = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()

            # Splited utterance matrix
            max_audio = eval_frames * 160 + 240 # max_frames = 300, frame_shift = 160, frame_length = 400
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = numpy.pad(audio, (0, shortage), 'wrap')
            feats = []
            startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf)+max_audio])
            feats = numpy.stack(feats, axis=0).astype(numpy.float64)
            data_2 = torch.FloatTensor(feats).cuda()
            # Speaker embeddings
            with torch.no_grad():
                embedding_1 = self.__model__(data_1)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                embedding_2 = self.__model__(data_2)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            embeddings[file] = [embedding_1, embedding_2]

        scores, labels = [], []

        for line in lines:
            embedding_11, embedding_12 = embeddings[line.split()[1]]
            embedding_21, embedding_22 = embeddings[line.split()[2]]

            # Compute the scores by matrix multiplication
            score_1 = torch.mean(torch.matmul(
                embedding_11, embedding_21.T))  # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))

            # compute score by cosine distance
            # score_1 = torch.mean(torch.cosine_similarity(embedding_11, embedding_21))
            # score_2 = torch.mean(torch.cosine_similarity(embedding_12, embedding_22))
            
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(int(line.split()[0]))

        return scores, labels

    def save_embeddings(self, test_list, test_path, eval_frames, model, output_filename_pk, output_filename_txt,output_path, **kwargs):
        # test_list_1 = "/kaggle/input/public-tst/example_submission/submission.txt"
        # test_path_1 = "/kaggle/input/public-tst"
        test_list_1 = test_list
        test_path_1 = test_path

        self.__model__.eval()

        nonorm_filename = output_filename_txt
        f_write_nonorm = open(os.path.join(output_path, nonorm_filename), "w")        

        files = []
        embeddings = {}
        embeddings_score = {}
        spk_emb_dic = {}  # save embeddings of an utterance
        lines = open(test_list_1).read().splitlines()
        for line in lines:
            files.append(line.split()[0]) #test
            files.append(line.split()[1]) #enr
        setfiles = list(set(files))
        setfiles.sort()
        # setfiles = ['wav/pr2.wav', 'wav/pr3.wav', 'wav/pr4.wav']      

        for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
            audio, _ = soundfile.read(os.path.join(test_path_1, file))
            # Full utterance
            data_1 = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()

            # Splited utterance matrix
            max_audio = eval_frames * 160 + 240
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = numpy.pad(audio, (0, shortage), 'wrap')
            feats = []
            startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf)+max_audio])
            feats = numpy.stack(feats, axis=0).astype(numpy.float64)
            data_2 = torch.FloatTensor(feats).cuda()
            # Speaker embeddings
            with torch.no_grad():
                embedding_1 = self.__model__(data_1)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                embedding_2 = self.__model__(data_2)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
                # concate embeddings of an utterance
                embedding_cc = torch.cat(
                    [embedding_1, embedding_2], dim=0).detach().cpu().numpy()
            embeddings_score[file] = [embedding_1, embedding_2]
            embeddings[file] = embedding_cc

        spk_emb_dic_enr = {}
        spk_emb_dic_test = {}
        
        for line in lines:
            spk_emb_dic_enr[line.split()[1]] = embeddings[line.split()[1]]
            spk_emb_dic_test[line.split()[0]] = embeddings[line.split()[0]]
            embedding_11, embedding_12 = embeddings_score[line.split()[0]]
            embedding_21, embedding_22 = embeddings_score[line.split()[1]]

            # # Compute the scores
            score_1 = torch.mean(torch.matmul(
                embedding_11, embedding_21.T))  # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()

            f_write_nonorm.write(line.split()[0] + '\t' +
                          line.split()[1] + '\t' + str(score) + '\n')
        f_write_nonorm.close()

        ASV_embedings_save_path = "save_emb"
        model_name = model
        os.makedirs(os.path.join(ASV_embedings_save_path, model_name), exist_ok=True)

        # data_dirnmae="enr"
        with open(f"{os.path.join(ASV_embedings_save_path, model_name)}/{model}_enr.pk", "wb") as f:
            pk.dump(spk_emb_dic_enr, f)
        
        # data_dirnmae="test"
        with open(f"{os.path.join(ASV_embedings_save_path, model_name)}/{model}_test.pk", "wb") as f:
            pk.dump(spk_emb_dic_test, f)

        return 0

    def save_embbeddings_2(self, test_list, test_path, eval_frames, model, output_filename_pk, **kwargs):
        test_list_1 = test_list
        test_path_1 = test_path
        
        self.__model__.eval()
        files = []
        embeddings = {}
        spk_emb_dic = {}  # save embeddings of an utterance
        lines = open(test_list_1).read().splitlines()
        # choose randomly 1200 lines
        # lines = random.sample(lines, 1200)     
        print(lines[0])   
        for line in lines:
            # files.append(line.split()[0]) #enr
            files.append(line.split()[1]) #train - enrollment
        setfiles = list(set(files))
        setfiles.sort()
        # setfiles = ['wav/pr2.wav', 'wav/pr3.wav', 'wav/pr4.wav']
        # Create a list to store the elements after '/'
        elements_after_slash = []
        it = 1

        # Iterate through each item in setfiles
        for file_path in setfiles:
            print(file_path)
            if it == 3:
                break
            else:
                it += 1
        print("hello\n")       

        for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
            audio, _ = soundfile.read(os.path.join(test_path_1, file))
            # Full utterance
            data_1 = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()

            # Splited utterance matrix
            max_audio = eval_frames * 160 + 240
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = numpy.pad(audio, (0, shortage), 'wrap')
            feats = []
            startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf)+max_audio])
            feats = numpy.stack(feats, axis=0).astype(numpy.float64)
            data_2 = torch.FloatTensor(feats).cuda()
            # Speaker embeddings
            with torch.no_grad():
                embedding_1 = self.__model__(data_1)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                embedding_2 = self.__model__(data_2)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
                # concate embeddings of an utterance
                embedding_cc = torch.cat(
                    [embedding_1, embedding_2], dim=0).detach().cpu().numpy()

            embeddings[file] = embedding_cc

        scores, labels = [], []
        # spk_emb_dic_enr = {}
        spk_emb_dic_test = {}
        for line in lines:
            # spk_emb_dic_enr[line.split()[0]] = embeddings[line.split()[0]]
            spk_emb_dic_test[line.split()[1]] = embeddings[line.split()[1]]
        
        ASV_embedings_save_path = "save_emb"
        model_name = model
        os.makedirs(os.path.join(ASV_embedings_save_path, model_name), exist_ok=True)

        # data_dirnmae="enr"
        # with open(f"{os.path.join(ASV_embedings_save_path, model_name)}/{data_dirnmae}_norm.pk", "wb") as f:
        #     pk.dump(spk_emb_dic_enr, f)
        
        data_dirnmae="train1200"
        with open(f"{os.path.join(ASV_embedings_save_path, model_name)}/{data_dirnmae}_{model}.pk", "wb") as f:
            pk.dump(spk_emb_dic_test, f)

        return 0
    
    def asnorm_score(params, selected_trials, embd_trn, train_cohort,asv_score, device="cuda"):
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

        # calculate norm score
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

    def save_score_as_norm(self, test_list, test_path, output_path,eval_frames, **kwargs):
        self.__model__.eval()
        files = []
        # filename = test_list.split("/")[-1]
        asnorm_filename = "test_list_asnorm.txt"
        nonorm_filename = "test_list_nonorm.txt"
        f_write_asnorm = open(os.path.join(output_path, asnorm_filename), "w")
        # f_write_nonorm = open(os.path.join(output_path, nonorm_filename), "w")

        embeddings = {}
        # public test: [test_1.wav  enr_1.wav   0.5]
        lines = open(test_list).read().splitlines()

        for line in lines:
            files.append(line.split()[0])
            files.append(line.split()[1])
        setfiles = list(set(files))
        setfiles.sort()

        for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
            audio, _ = soundfile.read(os.path.join(test_path, file))
            # Full utterance
            data_1 = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()

            # Splited utterance matrix
            max_audio = eval_frames * 160 + 240
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = numpy.pad(audio, (0, shortage), 'wrap')
            feats = []
            startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf)+max_audio])
            feats = numpy.stack(feats, axis=0).astype(numpy.float64)
            data_2 = torch.FloatTensor(feats).cuda()
            # Speaker embeddings
            with torch.no_grad():
                embedding_1 = self.__model__(data_1)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                embedding_2 = self.__model__(data_2)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            embeddings[file] = [embedding_1, embedding_2]

        for line in lines:
            embedding_11, embedding_12 = embeddings[line.split()[0]]
            embedding_21, embedding_22 = embeddings[line.split()[1]]

            # Compute the scores
            score_1 = torch.mean(torch.matmul(
                embedding_11, embedding_21.T))  # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))

            # # Compute score by cosine distance
            # score_1 = torch.mean(torch.cosine_similarity(embedding_11, embedding_21))
            # score_2 = torch.mean(torch.cosine_similarity(embedding_12, embedding_22))

            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()

            # compute asnorm score
            params={
                    "cohort_size": 1200
                }
            with open("/kaggle/working/random_train_norm.pk", "rb") as f:
                embd_trn = pk.load(f)
            N = 300
            selected_trials = list(embd_trn.keys())[:N * 2]
            tensor_list = [torch.FloatTensor(arr) for arr in embd_trn.values()]
            # Tính điểm số cho các cặp thử nghiệm và lưu vào một tensor
            scores2, norm_score = self.asnorm_score(params, selected_trials, embd_trn, tensor_list, score)

            f_write_asnorm.write(line.split()[0] + '\t' +
                          line.split()[1] + '\t' + str(norm_score) + '\n')
            # f_write_nonorm.write(line.split()[0] + '\t' +
            #                 line.split()[1] + '\t' + str(score) + '\n')
        f_write_asnorm.close()
        # f_write_nonorm.close()


    def save_score_no_norm(self, test_list, test_path, output_path,eval_frames,model, output_filename_txt, **kwargs):
        print("...saving score no norm...")
        self.__model__.eval()
        files = []
        nonorm_filename = output_filename_txt
        f_write_nonorm = open(os.path.join(output_path, nonorm_filename), "w")

        embeddings = {}
        lines = open(test_list).read().splitlines()

        for line in lines:
            files.append(line.split()[0]) # test
            files.append(line.split()[1]) # enr
        setfiles = list(set(files))
        setfiles.sort()

        for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
            audio, _ = soundfile.read(os.path.join(test_path, file))
            # Full utterance
            data_1 = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()

            # Splited utterance matrix
            max_audio = eval_frames * 160 + 240
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = numpy.pad(audio, (0, shortage), 'wrap')
            feats = []
            startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf)+max_audio])
            feats = numpy.stack(feats, axis=0).astype(numpy.float64)
            data_2 = torch.FloatTensor(feats).cuda()
            # Speaker embeddings
            with torch.no_grad():
                embedding_1 = self.__model__(data_1)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                embedding_2 = self.__model__(data_2)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            embeddings[file] = [embedding_1, embedding_2]

        for line in lines:
            embedding_11, embedding_12 = embeddings[line.split()[0]]
            embedding_21, embedding_22 = embeddings[line.split()[1]]

            # # Compute the scores
            score_1 = torch.mean(torch.matmul(
                embedding_11, embedding_21.T))  # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))

            # # Compute score by cosine distance
            # score_1 = torch.mean(torch.cosine_similarity(embedding_11, embedding_21))
            # score_2 = torch.mean(torch.cosine_similarity(embedding_12, embedding_22))
            
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()

            f_write_nonorm.write(line.split()[0] + '\t' +
                          line.split()[1] + '\t' + str(score) + '\n')
        f_write_nonorm.close()


    # ===== ===== ===== ===== ===== ===== ===== =====
    # Save parameters
    # ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):

        torch.save(self.__model__.module.state_dict(), path)
        # torch.save(self.__model__.__S__.state_dict(), path) # only save the model, not include the loss function model

    # ===== ===== ===== ===== ===== ===== ===== =====
    # Load parameters
    # ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.__model__.module.state_dict()
        loaded_state = torch.load(path, map_location="cuda:%d" % self.gpu)
        if len(loaded_state.keys()) == 1 and "model" in loaded_state:
            loaded_state = loaded_state["model"]
            newdict = {}
            delete_list = []
            for name, param in loaded_state.items():
                new_name = "__S__."+name
                newdict[new_name] = param
                delete_list.append(name)
            loaded_state.update(newdict)
            for name in delete_list:
                del loaded_state[name]
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")

                if name not in self_state:
                    print("{} is not in the model.".format(origname))
                    continue

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue

            self_state[name].copy_(param)
