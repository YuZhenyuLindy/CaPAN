# -*- coding: utf-8 -*

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb
from grl import WarmStartGradientReverseLayer, GradientReverseFunction
import mmd

def MI(outputs_target):
    batch_size = outputs_target.size(0)
    softmax_outs_t = nn.Softmax(dim=1)(outputs_target)
    avg_softmax_outs_t = torch.sum(softmax_outs_t, dim=0) / float(batch_size)
    log_avg_softmax_outs_t = torch.log(avg_softmax_outs_t + 1e-8)
    item1 = -torch.sum(avg_softmax_outs_t * log_avg_softmax_outs_t)
    item2 = -torch.sum(softmax_outs_t * torch.log(softmax_outs_t + 1e-8)) / float(batch_size)
    return item1, item2

def JS_Divergence_With_Temperature(p, q, temp_factor, get_softmax=True, l2_distance = False):
    KLDivLoss = nn.KLDivLoss(reduction='sum')
    if get_softmax:
        p_softmax_output = F.softmax(p / temp_factor)
        q_softmax_output = F.softmax(q / temp_factor)
    if l2_distance:
       return torch.mean(torch.abs(p_softmax_output - q_softmax_output))
    log_mean_softmax_output = ((p_softmax_output + q_softmax_output) / 2).log()
    return (KLDivLoss(log_mean_softmax_output, p_softmax_output) + KLDivLoss(log_mean_softmax_output, q_softmax_output)) / 2

def get_inter_pair_loss(labels_source, labels_target, outputs_source, outputs_target, temp_factor, threshold, l2_distance = False):
    loss = 0.0
    count = 0
    batch_size = len(labels_source)
    softmax_outs_target = nn.Softmax(dim=1)(outputs_target)
    for i in range(batch_size):
        for j in range(batch_size):
            if softmax_outs_target[j][labels_target[j]] < threshold:  # Threshold selection
                continue
            elif labels_source[i] == labels_target[j]:
                count += 1
                loss += JS_Divergence_With_Temperature(outputs_source[i], outputs_target[j], temp_factor, l2_distance=l2_distance)
    if count == 0:
        return loss
    else:
        return loss / count

def get_intra_pair_loss(labels, outputs, temp_factor, l2_distance = False):
    loss = 0.0
    count = 0
    batch_size = labels.size(0)
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            if labels[i] == labels[j]:
                count += 1
                loss += JS_Divergence_With_Temperature(outputs[i], outputs[j], temp_factor, l2_distance=l2_distance)
    if count == 0:
        return loss
    else:
        return loss / count

def get_PDD_loss(outs, labels, temp=1.0, threshold=0.8, l2_distance = False, factor=1.0):
    batch_size = outs.size(0) // 2
    batch_source = outs[: batch_size]
    batch_target = outs[batch_size:]
    labels_s = labels[:batch_size]
    labels_t = labels[batch_size:]
    loss_pdd_s_t = get_inter_pair_loss(labels_s, labels_t, batch_source, batch_target, temp, threshold, l2_distance)
    loss_pdd_s_s = get_intra_pair_loss(labels_s, batch_source, temp, l2_distance)
    
    return (temp ** 2) * (loss_pdd_s_s + loss_pdd_s_t) * factor 



class AdversarialLoss_PDD(nn.Module):
    def __init__(self, classifier: nn.Module):
        super(AdversarialLoss_PDD, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.classifier = classifier
        self.iters = 0

    def forward(self, f, labels_s, args, factor=1.0, stats=False):
        labels_s, labels_t = labels_s.chunk(2,dim=0)
        f_grl = self.grl(f)
        y = self.classifier(f_grl)
        
        self.iters = self.iters + 1

        temp = args.feature_temp
        if args.feature_normal:
            y = y / temp

        y_s, y_t = y.chunk(2, dim=0)

        pseudo_label_t = y_t.argmax(1)
        labels = torch.cat((labels_s, pseudo_label_t), dim=0)
        loss_pdd = get_PDD_loss(y, labels, temp=args.temp, threshold=args.threshold, l2_distance=args.l2_distance, factor=factor)

        entropy_target = torch.tensor(0.0).cuda()

        return loss_pdd, entropy_target
        
class AdversarialLoss_PDD_Double(nn.Module):
    def __init__(self, classifier: nn.Module, classifier_aux: nn.Module):
        super(AdversarialLoss_PDD_Double, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.classifier = classifier
        self.classifier_aux = classifier_aux
        self.iters = 0

    def forward(self, f, labels_s, args, factor=1.0):
        self.iters = self.iters + 1
        labels_s, labels_t = labels_s.chunk(2,dim=0)

        f_grl = self.grl(f)
        y = self.classifier(f_grl)
        y_s, y_t = y.chunk(2, dim=0)

        if args.feature_normal:
            f_grl_aux = nn.functional.normalize(f_grl)
            y_aux = self.classifier_aux(f_grl_aux) / args.feature_temp
        else:
            y_aux = self.classifier_aux(f_grl)
        y_s_aux, y_t_aux = y_aux.chunk(2, dim=0)
        
        source_logit = nn.Softmax(dim=1)(y_s )
        target_logit = nn.Softmax(dim=1)(y_t )
        source_logit_aux = nn.Softmax(dim=1)(y_s_aux)
        target_logit_aux = nn.Softmax(dim=1)(y_t_aux)

        pseudo_label_t = y_t.argmax(1)
        labels = torch.cat((labels_s, pseudo_label_t), dim=0)
        loss_pdd = torch.tensor(0.0).cuda() # loss_pdd = get_PDD_loss(y, labels, temp=args.temp, threshold=args.threshold, l2_distance=args.l2_distance, factor=factor)

        entropy_source = -args.entropy_tradeoff1 * torch.nn.functional.nll_loss(torch.log(source_logit_aux + 1e-5),labels_s)   
        if args.loss_type == 1:   # Class-aware Discriminator 1 : Using pseudo label on target domain from aux head
            pseudo_logit_t, pseudo_label_t = target_logit_aux.max(1)
            entropy_target = -torch.nn.functional.nll_loss(torch.log(1 - target_logit_aux + 1e-5), pseudo_label_t)
        elif args.loss_type == 2: # Class-aware Discriminator 2 : Using pseudo label on target domain from main head
            pseudo_logit_t, pseudo_label_t = target_logit.max(1)
            entropy_target = -torch.nn.functional.nll_loss(torch.log(1 - target_logit_aux + 1e-5), pseudo_label_t)
        elif args.loss_type == 3:
            binary_logit_t = torch.sigmoid(y_t_aux)
            entropy_target = target_logit.detach() * torch.log(1 - binary_logit_t + 1e-5)
            entropy_target = torch.mean(torch.sum(entropy_target, dim=1))
        elif args.loss_type == 4:                     # Class-aware Discriminator 3 : Rewighting without detach
            entropy_target = target_logit * torch.log(1 - target_logit_aux + 1e-5)
            entropy_target = torch.mean(torch.sum(entropy_target, dim=1))
        else:                     # Class-aware Discriminator 3 : Rewighting 
            entropy_target = target_logit.detach() * torch.log(1 - target_logit_aux + 1e-5)
            entropy_target = torch.mean(torch.sum(entropy_target, dim=1))

        loss_entropy = entropy_source + entropy_target
        return loss_pdd, loss_entropy
