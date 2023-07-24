import torch
import pickle as pl
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import Dataset, DataLoader
import random
from scipy.stats import pearsonr
import numpy as np

import warnings
warnings.filterwarnings("ignore")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



best_mean_pcc = 0.

count = 0

all_list = {}

def get_best(net, dev_dataloader, test_loader):
    global best_mean_pcc, all_list, count
    count = count + 1
    v_pcc, a_pcc = eval(dev_dataloader, net)
  
    if a_pcc + v_pcc > best_mean_pcc * 2.0:
        best_mean_pcc = (a_pcc + v_pcc)/2.0
        test_pcc_v, test_pcc_a = eval(dev_dataloader, net)
        all_list["best"] = [test_pcc_v, test_pcc_a]
        count = 0
    return count



class MyDataset(Dataset):
    def __init__(self, path, is_unlabeled=0.2, is_test=False, is_dev=False,  labeled_ratios = False):
        self.path = path
        self.all_data = pl.load(open(path, "rb"))
        random.shuffle(self.all_data)
        if is_unlabeled:
            self.all_data = self.all_data[int(len(self.all_data) * 0.4):int(len(self.all_data) * 0.9)]
            self.all_data = self.all_data[:int(len(self.all_data) * is_unlabeled)]
        if is_test:
            self.all_data = self.all_data[int(len(self.all_data) * 0.9):]
        if labeled_ratios:
            self.all_data = self.all_data[:int(len(self.all_data) * labeled_ratios)]
        if is_dev:
            self.all_data = self.all_data[int(len(self.all_data) * 0.5):int(len(self.all_data) * 0.7)]

    def __getitem__(self, index):
        return (self.all_data[index][0],self.all_data[index][1])

    def __len__(self):
        return len(self.all_data)


def get_dataloader(task, labeled_ratios=0.1, bs=64, is_unlabel=0.2):
    if task=="cvat":
        cvat_labeled = MyDataset("./data/processed_cvat.pkl", labeled_ratios=labeled_ratios)
        cvat_unlabeled = MyDataset("./data/processed_cvat.pkl", is_unlabeled=is_unlabel)
        cvat_test = MyDataset("./data/processed_cvat.pkl", is_test=True)
        cvat_dev = MyDataset("./data/processed_cvat.pkl", is_dev=True)

        cvat_labeled_loader = DataLoader(dataset=cvat_labeled,batch_size=bs,shuffle=True,collate_fn = collate_fn, num_workers=0)
        cvat_unlabeled_loader = DataLoader(dataset=cvat_unlabeled,batch_size=bs,shuffle=True,collate_fn = collate_fn,num_workers=0)
        cvat_test_loader = DataLoader(dataset=cvat_test,batch_size=bs,shuffle=True,collate_fn = collate_fn,num_workers=0)
        cvat_dev_loader = DataLoader(dataset=cvat_dev,batch_size=bs,shuffle=True,collate_fn = collate_fn,num_workers=0)
        return cvat_labeled_loader, cvat_unlabeled_loader, cvat_test_loader, cvat_dev_loader

    if task=="fb":
        fb_labeled = MyDataset("./data/fb.pkl", labeled_ratios=labeled_ratios)
        fb_unlabeled = MyDataset("./data/fb.pkl", is_unlabeled=is_unlabel)
        fb_test = MyDataset("./data/fb.pkl", is_test=True)
        fb_dev = MyDataset("./data/fb.pkl", is_dev=True)

        fb_labeled_loader = DataLoader(dataset=fb_labeled,batch_size=bs,shuffle=True,collate_fn = collate_fn, num_workers=0)
        fb_unlabeled_loader = DataLoader(dataset=fb_unlabeled,batch_size=bs,shuffle=True,collate_fn = collate_fn,num_workers=0)
        fb_test_loader = DataLoader(dataset=fb_test,batch_size=bs,shuffle=True,collate_fn = collate_fn,num_workers=0)
        fb_dev_loader = DataLoader(dataset=fb_dev,batch_size=bs,shuffle=True,collate_fn = collate_fn,num_workers=0)
        return fb_labeled_loader, fb_unlabeled_loader, fb_test_loader, fb_dev_loader

    if task=="emobank":
        emobank_labeled = MyDataset("./data/emobank.pkl", labeled_ratios=labeled_ratios)
        emobank_unlabeled = MyDataset("./data/emobank.pkl", is_unlabeled=is_unlabel)
        emobank_test = MyDataset("./data/emobank.pkl", is_test=True)
        emobank_dev = MyDataset("./data/emobank.pkl", is_dev=True)
        emobank_labeled_loader = DataLoader(dataset=emobank_labeled,batch_size=bs,shuffle=True,collate_fn = collate_fn, num_workers=0)
        emobank_unlabeled_loader = DataLoader(dataset=emobank_unlabeled,batch_size=bs,shuffle=True,collate_fn = collate_fn,num_workers=0)
        emobank_test_loader = DataLoader(dataset=emobank_test,batch_size=bs,shuffle=True,collate_fn = collate_fn,num_workers=0)
        emobank_dev_loader = DataLoader(dataset=emobank_dev,batch_size=bs,shuffle=True,collate_fn = collate_fn,num_workers=0)
        return emobank_labeled_loader, emobank_unlabeled_loader, emobank_test_loader, emobank_dev_loader

def get_first(x):
    return x[0]

def get_last(x):
    return x[-1]

def collate_fn(batch):
    x = list(map(get_first,batch))
    y = list(map(get_last,batch))
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    return x, y




class BatchFormer(nn.Module):
    def __init__(self):
        super(BatchFormer, self).__init__()

        # d_model – the number of expected features in the input (required).
        # nhead – the number of heads in the multiheadattention models (required).
        # dim_feed - forward – the dimension of the feedforward network model (default=2048).
        # dropout – the dropout value (default=0.1).
        self.encoder = torch.nn.TransformerEncoderLayer(768, 8, 768, 0.5) 

    def forward(self, x):
        x = self.encoder(x.unsqueeze(1)).squeeze(1)
        return x
        
        
class BiLSTM(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=768, out_dim=768):
        super(BiLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True, dropout=0.5)
        self.fc_out = nn.Linear(hidden_dim, self.out_dim)
        
    def forward(self, embeds):
        lstm_out, _ = self.lstm(embeds)
        return lstm_out





 
class MyNet(nn.Module):
    def __init__(self, relation_model=None, noise=0.05):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(768, 768)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(768, 2)
        self.relation_model = relation_model
        if relation_model is not None:
            self.relation_model = relation_model
            self.fc2 = nn.Linear(768 * 2, 768)
 
    def forward(self, x, is_unlabel=False):

        if self.training:
            noise1 = torch.normal(0,noise,size=x.shape).cuda()
            x = x + noise1
        x = self.fc1(x)
        x = self.relu1(x)

        if self.relation_model is not None:
            if self.training:
                noise2 = torch.normal(0,noise,size=x.shape).cuda()
                x = x + noise2
            r_x = self.relation_model(x)
            x = torch.concat([x, r_x], dim = -1)
            x = self.fc2(x)
        x = self.out(x)
        return x

class BERT(nn.Module):
    def __init__(self, relation_model=None):
        super(BERT, self).__init__()
        self.fc = nn.Linear(768, 512)
        self.out = nn.Linear(512, 2)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        out = self.out(x)
        return out

def eval(dataloader, model):
    model.eval()
    pred_v_list = []
    pred_a_list = []
    real_v_list = []
    real_a_list = []
    for x, y in dataloader:
        pred_y = model(x.cuda())

        pred_v_list.extend(list(map(get_first,pred_y.cpu().tolist())))
        pred_a_list.extend(list(map(get_last,pred_y.cpu().tolist())))

        real_v_list.extend(list(map(get_first,y.cpu().tolist())))
        real_a_list.extend(list(map(get_last,y.cpu().tolist())))

    pcc_v = pearsonr(pred_v_list, real_v_list)
    pcc_a = pearsonr(pred_a_list, real_a_list)
    model.train()

    return pcc_v[0], pcc_a[0]

    




def train_bert(epochs=1200, task="cvat", labeled_ratios=0.1, bs=64, exp_name="no_name"):
    
    loss_fun = torch.nn.MSELoss(reduction='mean')

    net = BERT().cuda(0)
    opt = torch.optim.Adam(net.parameters(), lr = 1e-4)

    labeled_loader, unlabeled_loader, test_loader, dev_loader =  get_dataloader(task, labeled_ratios, bs=bs, is_unlabel=0.2)
    

    for epoch in range(epochs):
        for x, y in labeled_loader:
            
            pred_y1 = net(x.cuda())
            loss_labeled = (loss_fun(pred_y1, y.cuda()))
            pred_y2 = net(x.cuda())
            loss_labeled = (loss_fun(pred_y1, y.cuda()) + loss_fun(pred_y2, y.cuda())) * 0.5 + loss_fun(pred_y1, pred_y2)

            ux, _ = next(iter(unlabeled_loader))
            pred_uy_1 = net(ux.cuda())
            pred_uy_2 = net(ux.cuda())
            loss_unlabel = loss_fun(pred_uy_1, pred_uy_2)

            loss = loss_labeled + loss_unlabel

            opt.zero_grad()
            loss.backward()
            opt.step()

        count = get_best(net, dev_loader, test_loader)

        if count > 10:
            for key in all_list:
                print(all_list[key])
                print()
            exit()


            



def train(epochs=1200, task="cvat", labeled_ratios=0.1, submodel=BiLSTM(), bs=64, noise=0.05, exp_name="no_name", is_unlabel):
    
    loss_fun = torch.nn.MSELoss(reduction='mean')

    net = MyNet(None).cuda(0)
    if submodel is not None:
        net = MyNet(submodel.cuda(), noise=noise).cuda(0)

    opt = torch.optim.Adam(net.parameters(), lr = 1e-4)

    labeled_loader, unlabeled_loader, test_loader, dev_loader =  get_dataloader(task, labeled_ratios, bs=bs, is_unlabel=0.2)

    for epoch in range(epochs):
        losses =[]
        for ux, _ in unlabeled_loader:

            pred_uy_1 = net(ux.cuda())
            pred_uy_2 = net(ux.cuda())
            loss_unlabel = loss_fun(pred_uy_1, pred_uy_2)
            
            x, y = next(iter(labeled_loader))
            pred_y1 = net(x.cuda())
            loss_labeled = (loss_fun(pred_y1, y.cuda()))
            pred_y2 = net(x.cuda())
            loss_labeled = (loss_fun(pred_y1, y.cuda()) + loss_fun(pred_y2, y.cuda())) * 0.5 + loss_fun(pred_y1, pred_y2)
            
            loss = loss_labeled + loss_unlabel

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.detach().cpu())

        count = get_best(net, dev_loader, test_loader)

        if count > 10:
            f = open("./"+str(exp_name)+".txt", "a+")

            for key in all_list:
                print(all_list[key])
                f.writelines(str(all_list[key])+"\n")
                print()
            f.close()
            exit()

    



from sys import argv


if __name__ == '__main__':
    _, task, labeled_ratios, seed, submodel, epochs, bs, noise, exp_name, is_unlabel = argv

    labeled_ratios = float(labeled_ratios)
    bs = int(bs)
    noise = float(noise)
    epochs = int(epochs)
    exp_name = str(exp_name)
    is_unlabel = float(is_unlabel)


    setup_seed(int(seed))
    if submodel=="lstm":
        train(epochs=epochs, task=task, labeled_ratios=labeled_ratios, submodel=BiLSTM(), bs=bs, noise=noise, exp_name=exp_name)
    if submodel=="transformer":
        train(epochs=epochs, task=task, labeled_ratios=labeled_ratios, submodel=BatchFormer(), bs=bs, noise=noise, exp_name=exp_name, is_unlabel=is_unlabel)
    if submodel == "none":
        train(epochs=epochs, task=task, labeled_ratios=labeled_ratios, submodel=None, bs=bs, noise=noise, exp_name=exp_name)
    if submodel == "bert":
        train_bert(epochs=epochs, task=task, labeled_ratios=labeled_ratios, bs=bs, exp_name=exp_name)
    

