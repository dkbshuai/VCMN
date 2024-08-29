import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os.path as osp
import utils
from utils import AverageMeter
import MLdataset
import argparse
import time
from MLPMixer import get_model
import torch
import numpy as np
from myloss import Loss
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
import copy
import tqdm
from measure import *
# import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

def wmse_loss(input, target, weight, reduction='mean'):
    ret = (torch.diag(weight).mm(target - input)) ** 2
    ret = torch.mean(ret)
    return ret


def do_metric(y_prob, label):

    y_predict = y_prob > 0.5
    ranking_loss = 1 - compute_ranking_loss(y_prob, label)
    # print(ranking_loss)
    one_error = 1 - compute_one_error(y_prob, label)
    # print(one_error)
    coverage = 1 - compute_coverage(y_prob, label)
    # print(coverage)
    hamming_loss = 1 - compute_hamming_loss(y_predict, label)
    # print(hamming_loss)
    precision = compute_average_precision(y_prob, label)
    # print(precision)
    macro_f1 = compute_macro_f1(y_predict, label)
    # print(macro_f1)
    micro_f1 = compute_micro_f1(y_predict, label)
    # print(micro_f1)
    auc = compute_auc(y_prob, label)
    auc_me = mlc_auc(y_prob, label)
    return np.array([precision, hamming_loss, ranking_loss, auc_me, one_error, coverage, auc, macro_f1, micro_f1])

"""main train"""

train_loss_list = []
train_accurate_list = []
val_accurate_list = []
test_accurate_list = []

def train(loader, model, loss_model, optimizer, sche, epoch,logger):

    model.train()
    
    train_steps = len(loader)
    train_loss = 0.
    total_labels = []
    total_preds = []
    individual_all_z = []
    
    #loader = tqdm(loader)
    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        
        data=[v_data.to('cuda:0') for v_data in data]
        label = label.to('cuda:0')
        inc_V_ind = inc_V_ind.float().to('cuda:0')
        inc_L_ind = inc_L_ind.float().to('cuda:0')
        
        pred_x, x_cont = model(data, inc_V_ind)

        loss_Cont = 0
        if args.view_loss:
                loss_Cont = loss_model.contrastive_loss(x_cont,label.cuda(),inc_V_ind.cuda(), inc_L_ind.cuda())
                
        loss_CL = torch.mean(torch.abs((label.mul(torch.log(pred_x + 1e-10)) \
                                            + (1-label).mul(torch.log(1 - pred_x + 1e-10)))\
                                       #))
                                       .mul(inc_L_ind)))
                
        fusion_loss = loss_CL + loss_Cont * args.beta 
    
        train_loss += fusion_loss
        optimizer.zero_grad()
        fusion_loss.backward()
        optimizer.step()
        loader.desc = "train epoch[{}/{}]".format(epoch+1, args.epochs)
        pred = pred_x.cpu()
        total_labels = np.concatenate((total_labels,label.cpu().numpy()),axis=0) if len(total_labels)>0 else label.cpu().numpy()
        total_preds = np.concatenate((total_preds,pred.detach().numpy()),axis=0) if len(total_preds)>0 else pred.detach().numpy()

    if args.sche:
        sche.step()
        
    total_labels=np.array(total_labels)
    total_preds=np.array(total_preds)
    evaluation_results=do_metric(total_preds,total_labels)
        
    train_loss_list.append(train_loss.item()/train_steps)    
    train_accurate_list.append(evaluation_results[0].item())
    
    logger.info('Epoch:[{0}]\t'
                'train\t'
                  'Train Loss: {train_loss:.4f}\t\t'
                  'Train AP: {ap:.4f}\t'.format(
                        epoch+1,  
                        train_loss=train_loss.item()/train_steps,
                        ap=evaluation_results[0]))
    
    
    return fusion_loss, model, train_loss_list, train_accurate_list

def test(loader, model, loss_model, epoch,logger):
    total_labels = []
    total_preds = []
    model.eval()
    #loader = tqdm(loader)
    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        data=[v_data.to('cuda:0') for v_data in data]
        inc_V_ind = inc_V_ind.float().to('cuda:0')
        pred, _ = model(data,inc_V_ind)
        pred = pred.cpu()
        total_labels = np.concatenate((total_labels,label.numpy()),axis=0) if len(total_labels)>0 else label.numpy()
        total_preds = np.concatenate((total_preds,pred.detach().numpy()),axis=0) if len(total_preds)>0 else pred.detach().numpy()
        
    total_labels=np.array(total_labels)
    total_preds=np.array(total_preds)

    evaluation_results=do_metric(total_preds,total_labels)
    
    val_accurate_list.append(evaluation_results[0].item())
    
    logger.info('Epoch:[{0}]\t'
                'val\t\t' 
                  'AUC: {auc:.4f}\t'
                  'HL: {hl:.4f}\t'
                  'RL: {rl:.4f}\t'
                  'AP: {ap:.4f}\t\n'.format(
                        epoch+1,
                        auc=evaluation_results[3], 
                        hl=evaluation_results[1],
                        rl=evaluation_results[2],
                        ap=evaluation_results[0]
                        ))
    return evaluation_results


def main(args,file_path):
    data_path = osp.join(args.root_dir, args.dataset, args.dataset+'_six_view.mat')
    fold_data_path = osp.join(args.root_dir, args.dataset, args.dataset+'_six_view_MaskRatios_' + str(
                                args.mask_view_ratio) + '_LabelMaskRatio_' +
                                str(args.mask_label_ratio) + '_TraindataRatio_' + 
                                str(args.training_sample_ratio) + '.mat')
    
    folds_num = args.folds_num
    folds_results = [AverageMeter() for i in range(9)]
    if args.logs:
        logfile = osp.join(args.logs_dir,args.name+args.dataset+'_V_' + str(
                                    args.mask_view_ratio) + '_L_' +
                                    str(args.mask_label_ratio) + '_T_' + 
                                    str(args.training_sample_ratio) + '_'+str(args.alpha)+'_'+str(args.beta)+'.txt')
    else:
        logfile=None
    logger = utils.setLogger(logfile)
    device = torch.device('cuda:0')
    for fold_idx in range(folds_num):
        fold_idx=fold_idx
        train_dataloder,train_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,fold_idx=fold_idx,mode='train',batch_size=args.batch_size,shuffle = False,num_workers=4)
        test_dataloder,test_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,val_ratio=0.15,fold_idx=fold_idx,mode='test',batch_size=args.batch_size,num_workers=4)
        val_dataloder,val_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,fold_idx=fold_idx,mode='val',batch_size=args.batch_size,num_workers=4)
        d_list = train_dataset.d_list
        classes_num = train_dataset.classes_num
        
        
        model = get_model(len(d_list), d_list, num_classes=train_dataset.classes_num,embed_dim=512, depth=args.depth,\
                          view_inner_dim=64, expansion_factor=2, \
                              dropout=0.2, exponent=2).to(device) # view_inner_dim=32
        
        # print(model)
        loss_model = Loss(args.alpha,device)
        #optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
        optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.05)
        #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100)
        scheduler = CosineAnnealingLR(optimizer, args.epochs)
        #scheduler = None

        logger.info('train_data_num:'+str(len(train_dataset))+
                    '  val_data_num:'+str(len(val_dataset))+
                    '  test_data_num:'+str(len(test_dataset))+
                    '   fold_idx:'+str(fold_idx+1)+
                    '\nlr:'+str(args.lr)+
                    '  alpha:'+str(args.alpha)+
                    '  beta:'+str(args.beta)+
                    '  gamma:'+str(args.gamma))
        
        print("")
        print(args)
        print("")
        
        static_res = 0
        epoch_results = [AverageMeter() for i in range(9)]
        total_losses = AverageMeter()
        train_losses_last = AverageMeter()
        best_epoch=0
        best_model_dict = {'model':model.state_dict(),'epoch':0}
        for epoch in range(args.epochs):
            
            train_losses, model, train_loss_list,train_accurate_list= train(train_dataloder,model,loss_model,optimizer,scheduler,epoch,logger)
            val_results = test(val_dataloder,model,loss_model,epoch,logger)

            
            if val_results[0]*0.5+val_results[2]*0.25+val_results[3]*0.25>=static_res:
                static_res = val_results[0]*0.5+val_results[2]*0.25+val_results[3]*0.25
                #返回模型的状态字典。模型的状态字典是一个Python字典，其中包含了模型所有参数的当前权重和偏置。
                best_model_dict['model'] = copy.deepcopy(model.state_dict())
                best_model_dict['epoch'] = epoch
                best_epoch=epoch
            # if epoch >= 20 and ( epoch_results[0].max<0.20 or (epoch_results[0].max-val_results[0]>0.01) or (abs(train_losses_last.sum - train_losses.sum) < 1e-5)):
            #     logger.info('Early stop!: epoch=%d, best:epoch=%d, best_AP=%.7f,total_loss=%.7f' % (
            #         epoch, static_res.max_ind, epoch_results[0].vals[static_res.max_ind], train_losses.sum))
            #     break
            train_losses_last = train_losses
        model.load_state_dict(best_model_dict['model'])
        test_results = test(test_dataloder,model,loss_model,epoch,logger)

        logger.info('final: fold_idx:{} best_epoch:{}\t best:ap:{:.4}\t HL:{:.4}\t RL:{:.4}\t AUC_me:{:.4}\n'.format(fold_idx+1,best_epoch+1,test_results[0],test_results[1],
            test_results[2],test_results[3]))
        
        test_accurate_list.append(test_results[0].item())
        print('test_accurate_list:', test_accurate_list)
        print("")
        
        #保存9个评估结果及超参数
        for i in range(9):
            folds_results[i].update(test_results[i])
        if args.save_curve:
            np.save(osp.join(args.curve_dir,args.dataset+'_V_'+str(args.mask_view_ratio)+'_L_'+str(args.mask_label_ratio))+'_'+str(fold_idx)+'.npy', np.array(list(zip(epoch_results[0].vals,train_losses.vals))))
    file_handle = open(file_path, mode='a')
    if os.path.getsize(file_path) == 0:
        file_handle.write(
            'AP HL RL AUCme one_error coverage macAUC macro_f1 micro_f1 lr alpha beta gamma\n')
    # generate string-result of 9 metrics and two parameters
    res_list = [str(round(res.avg,4))+'+'+str(round(res.std,4)) for res in folds_results]
    res_list.extend(['lr='+str(args.lr),'alpha='+str(args.alpha),'beta='+str(args.beta)])
    res_list.extend(['datasets='+str(args.datasets),'epochs='+str(args.epochs),'batch_size='+str(args.batch_size),'sche='+str(args.sche),\
                     'model_type='+str(args.model_type),'depth='+str(args.depth),'view_loss='+str(args.view_loss)])
    res_str = ' '.join(res_list)
    file_handle.write(res_str)
    file_handle.write('\n')
    file_handle.close()
        

def filterparam(file_path,index):
    params = []
    if os.path.exists(file_path):
        file_handle = open(file_path, mode='r')
        lines = file_handle.readlines()
        lines = lines[1:] if len(lines)>1 else []
        params = [[float(line.split(' ')[idx]) for idx in index] for line in lines ]
    return params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # misc
    working_dir = osp.dirname(osp.abspath(__file__)) 
    parser.add_argument('--logs-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--logs', default=False, type=bool)
    parser.add_argument('--records-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'records'))
    parser.add_argument('--file-path', type=str, metavar='PATH', 
                        default='')
    parser.add_argument('--root-dir', type=str, metavar='PATH', 
                        default='./data/')
    
    parser.add_argument('--dataset', type=str, default='')# mirflickr corel5k pascal07 iaprtc12 espgame 
    parser.add_argument('--datasets', type=list, default=['corel5k'])
    parser.add_argument('--mask-view-ratio', type=float, default=0.5)
    parser.add_argument('--mask-label-ratio', type=float, default=0.5)
    parser.add_argument('--training-sample-ratio', type=float, default=0.7)
    parser.add_argument('--folds-num', default=20, type=int)
    parser.add_argument('--weights-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'weights'))
    parser.add_argument('--curve-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'curves'))
    parser.add_argument('--save-curve', default=False, type=bool)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--name', type=str, default='20_')
    parser.add_argument('--model_type', type=str, default=' v-d ')
    parser.add_argument('--view_loss', type=bool, default=True)  #  True  False
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--epochs', type=int, default= 15 )  # 3 for pascal07, 8 for iaprtc12, 15 for other
    parser.add_argument('--sche', type=bool, default=False)  #  True  False
    parser.add_argument('--lr', type=float, default=1e-4)  # not here, set it below
    parser.add_argument('--T_max', type=float, default=200)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=1e-1)  # not here, set it below
    parser.add_argument('--beta', type=float, default=1e-1)  # not here, set it below
    parser.add_argument('--gamma', type=float, default=1e-1)


    
    args = parser.parse_args()
    
    if args.logs:
        if not os.path.exists(args.logs_dir):
            os.makedirs(args.logs_dir)
    if args.save_curve:
        if not os.path.exists(args.curve_dir):
            os.makedirs(args.curve_dir)
    if True:
        if not os.path.exists(args.records_dir):
            os.makedirs(args.records_dir)
    lr_list = [0.0003]  # 0.0003
    alpha_list = [0.5]  # 0.9 for iaprtc12, 0.5 for other
    beta_list = [0.001]  #
    gamma_list = [1]

    
    for lr in lr_list:
        args.lr = lr
        if args.lr >= 0.01:
            args.momentumkl = 0.90
        for alpha in alpha_list:
            args.alpha = alpha
            for beta in beta_list:
                args.beta = beta
                for gamma in gamma_list:
                    args.gamma = gamma
                    for dataset in args.datasets:
                        args.dataset = dataset
                        file_path = osp.join(args.records_dir,args.name+args.dataset+'_ViewMask_' + str(
                                        args.mask_view_ratio) + '_LabelMask_' +
                                        str(args.mask_label_ratio) + '_Training_' + 
                                        str(args.training_sample_ratio) + '_bs128.txt')
                        args.file_path = file_path
                        existed_params = filterparam(file_path,[-3,-2,-1])
                        if [args.alpha,args.beta,args.gamma] in existed_params:
                            print('existed param! alpha:{} beta:{} gamma:{} '.format(args.alpha,args.beta,args.gamma))
                            continue
                        main(args,file_path)