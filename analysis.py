import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def collect_feature(data_loader: DataLoader, feature_extractor: nn.Module,
                    device: torch.device, max_num_features=None) -> torch.Tensor:
    """
    Fetch data from `data_loader`, and then use `feature_extractor` to collect features

    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        feature_extractor (torch.nn.Module): A feature extractor.
        device (torch.device)
        max_num_features (int): The max number of features to return

    Returns:
        Features in shape (min(len(data_loader), max_num_features * mini-batch size), :math:`|\mathcal{F}|`).
    """
    feature_extractor.eval()
    all_features = []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm.tqdm(data_loader)):
            if max_num_features is not None and i >= max_num_features:
                break
            images = images.to(device)
            _, feature = feature_extractor(images)
            feature = feature.cpu()
            all_features.append(feature)
    return torch.cat(all_features, dim=0)

def collect_output(data_loader: DataLoader, output_extractor: nn.Module,
                    device: torch.device, max_num_outputs=None, use_softmax=False) -> torch.Tensor:
    """
    Fetch data from `data_loader`, and then use `output_extractor` to collect features

    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        output_extractor (torch.nn.Module): A output extractor.
        device (torch.device)
        max_num_outputs (int): The max number of outputs to return

    Returns:
        Outputs in shape (min(len(data_loader), max_num_outputs * mini-batch size), :math:`|\mathcal{F}|`).
    """
    output_extractor.eval()
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm.tqdm(data_loader)):
            if max_num_outputs is not None and i >= max_num_outputs:
                break
            images = images.to(device)
            output,_ = output_extractor(images)
            output = output.cpu()
            if use_softmax:
                output = torch.nn.functional.softmax(output, dim=1)
            all_outputs.append(output)
            all_labels.append(target)
    return torch.cat(all_outputs, dim=0), torch.cat(all_labels, dim=0)

def collect_feature_output(data_loader: DataLoader, classifier: nn.Module,
                    device: torch.device, max_num_outputs=None, use_softmax=False) -> torch.Tensor:
    """
    Fetch data from `data_loader`, and then use `output_extractor` to collect features

    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        output_extractor (torch.nn.Module): A output extractor.
        device (torch.device)
        max_num_outputs (int): The max number of outputs to return

    Returns:
        Outputs in shape (min(len(data_loader), max_num_outputs * mini-batch size), :math:`|\mathcal{F}|`).
    """
    classifier.eval()
    all_features = []
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm.tqdm(data_loader)):
            if max_num_outputs is not None and i >= max_num_outputs:
                break
            images = images.to(device)
            
            feature = classifier.backbone(images)
            feature = feature.view(-1, classifier.backbone.out_features)
            feature = classifier.bottleneck(feature)
            output = classifier.head(feature)
            if use_softmax:
                output = torch.nn.functional.softmax(output, dim=1)
            # 修改后
            feature = torch.nn.functional.normalize(feature, dim=1)
            feature = feature.cpu()
            output = output.cpu()
            all_features.append(feature)
            all_outputs.append(output)
            all_labels.append(target)
    return torch.cat(all_features, dim=0), torch.cat(all_outputs, dim=0), torch.cat(all_labels, dim=0)

def plot_hist_outputs(source_output, source_label, target_output, target_label, save_filename, bins=50, use_ground_truth=True, density=False):

    num_classes = source_output.size()[1]
    if use_ground_truth == False:
        source_label = source_output.max(1)[1]
        target_label = target_output.max(1)[1]

    source_output = source_output.numpy()
    source_label = source_label.numpy()

    target_output = target_output.numpy()
    target_label = target_label.numpy()

    plt.figure(figsize=(8,8))
    
    plt.hist(np.concatenate([source_output[np.where(source_label==c),c][0] for c in range(num_classes)]), bins, alpha=0.5, label='Source Domain', density=density)
    plt.hist(np.concatenate([target_output[np.where(target_label==c),c][0] for c in range(num_classes)]), bins, alpha=0.5, label='Target Domain', density=density)
    plt.xticks(fontproperties='Times New Roman', size=16)
    plt.yticks(fontproperties='Times New Roman', size=16)
    plt.xlabel('The Classifier Output', fontsize=16)
    if density:
        plt.ylabel('Density', fontsize=16)
        plt.legend(loc='best', fontsize=16)
        # plt.legend(loc='upper center')
    else:
        plt.ylabel('Counts',fontsize=16)
        plt.legend(loc='upper left',fontsize=16)
    plt.savefig(save_filename)
    plt.close()

def plot_hist_twoclasses_outputs(source_output, source_label, target_output, target_label, save_filename, bins=50, density=False, show_all=True):

    num_classes = source_output.size()[1]
    pseudo_source_label = source_output.max(1)[1]
    pseudo_target_label = target_output.max(1)[1]

    source_output = source_output.numpy()
    source_label = source_label.numpy()
    pseudo_source_label = pseudo_source_label.numpy()

    target_output = target_output.numpy()
    target_label = target_label.numpy()
    pseudo_target_label = pseudo_target_label.numpy()
    
    source_correct_mask = np.where(source_label==pseudo_source_label)
    source_incorrect_mask = np.where(source_label!=pseudo_source_label)
    target_correct_mask = np.where(target_label==pseudo_target_label)
    target_incorrect_mask = np.where(target_label!=pseudo_target_label)

    if show_all:
        plt.hist(source_output[source_correct_mask, source_label[source_correct_mask]][0], bins, alpha=0.5, label='Source Correct Classes', density=density)
        plt.hist(source_output[source_incorrect_mask, pseudo_source_label[source_incorrect_mask]][0], bins, alpha=0.5, label='Source Incorrect Classes', density=density)
    plt.hist(target_output[target_correct_mask, target_label[target_correct_mask]][0], bins, alpha=0.5, label='Target Correct Classes', density=density)
    plt.hist(target_output[target_incorrect_mask, pseudo_target_label[target_incorrect_mask]][0], bins, alpha=0.5, label='Target Incorrect Classes', density=density)

    plt.xlabel('The Classifier Output')
    if density:
        plt.ylabel('Density')
        plt.legend(loc='best')
        # plt.legend(loc='upper center')
    else:
        plt.ylabel('Counts')
        plt.legend(loc='upper left')
    plt.savefig(save_filename)
    plt.close()


def accuracy_by_threshold(output, label):

    logit, pseudo_label = output.max(1)
    
    threshold = np.arange(0,1.1,0.1)
    accuracy_by_threshold = []

    for t in range(len(threshold)-1):
        t_min = threshold[t]
        t_max = threshold[t+1]
        mask_lower = logit > t_min
        mask_upper = logit <= t_max
        mask = mask_lower == mask_upper
        correct = pseudo_label[mask] == label[mask]
        if torch.sum(mask) > 0:
            accuracy = torch.sum(correct) / torch.sum(mask)
        else:
            accuracy = torch.tensor(0.0)
        accuracy_by_threshold.append(accuracy.item())

    return accuracy_by_threshold

    
def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()
    
def plot_confusion_matrix(cm, classes, png_output=None, normalize=False, cmap=plt.cm.Blues, show=False):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title='Normalized confusion matrix'
        else:
            title='Confusion matrix'

        # Calculate chart area size
        leftmargin = 0.5 # inches
        rightmargin = 0.5 # inches
        categorysize = 0.5 # inches
        figwidth = leftmargin + rightmargin + (len(classes) * categorysize)           

        f = plt.figure(figsize=(figwidth, figwidth))

        # Create an axes instance and ajust the subplot size
        ax = f.add_subplot(111)
        ax.set_aspect(1)
        f.subplots_adjust(left=leftmargin/figwidth, right=1-rightmargin/figwidth, top=0.94, bottom=0.1)

        res = ax.imshow(cm, interpolation='nearest', cmap=cmap)

        plt.title(title)
        plt.colorbar(res)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_yticklabels(classes)

        cm[:] = cm[:] * 100 # percent (%)
        fmt = '.1f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        # plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        if png_output is not None:
            os.makedirs(png_output, exist_ok=True)
            f.savefig(os.path.join(png_output,'confusion_matrix.png'), bbox_inches='tight')

        if show:
            plt.show()
            plt.close(f)
        else:
            plt.close(f)

def get_confusion_matrix(target_val_loader, model, class_num, use_cuda=True):
    model.eval()
    conf_matrix = torch.zeros(class_num,class_num)
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm.tqdm(target_val_loader)):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output, _ = model(data)
            pred = output.data.max(1)[1] # get the index of the max log-probability
            for t,p in zip(target, pred):
                conf_matrix[t,p] += 1
    return conf_matrix

# Test for Confuse Matrix
#     if is_confuse_matrix:
#         model.load_state_dict(best_model)
#         confusion_matrix = get_confusion_matrix(target_test_loader, model, class_num)
#         confusion_matrix = confusion_matrix.cpu().data.numpy()
#         plot_confusion_matrix(confusion_matrix,target_test_loader.dataset.classes,
#                               normalize=True)

# Print Accuracy for Per-Class
# acc_global = torch.sum(torch.diag(cm)) / torch.sum(cm)
# acc_per_class = torch.diag(cm) / torch.sum(cm, 1)
# print("Accuracy for Global: {}\n Accuracy for Per-Class: {}\n".format(acc_global.item() * 100, ['{:.2f}'.format(i) for i in (acc_per_class * 100).tolist()]))









