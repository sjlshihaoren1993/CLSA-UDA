import sys
import os.path as osp
import time
import argparse
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append('../')
from dalib.adaptation.cdan import ImageClassifier
import common.vision.datasets as datasets
import common.vision.models as models
from common.utils.metric import accuracy, ConfusionMatrix
from common.utils.meter import AverageMeter, ProgressMeter


def get_model(model_name, pretrain=False):
    if model_name in models.__dict__:
        # load models from common.vision.models
        backbone = models.__dict__[model_name](pretrained=pretrain)
    elif model_name == 'ResNet18_3D':
        backbone = models.resnet3d.ResNet18_3D(num_classes=2)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=pretrain)
        try:
            #backbone.out_features = backbone.get_classifier().in_features
            backbone.out_features = 768 
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    return backbone


def get_dataset(root):
    train_source_dataset = datasets.MRIDataset(
        root_dir_fa=osp.join(root, 'ppmi1'),
        labels_file='../../../data/ppmi1_shuffle.txt',  # 共享标签文件
        return_filenames=False
    )
    train_target_dataset =  datasets.MRIDataset(
        root_dir_fa=osp.join(root, 'hnph'),
        labels_file='../../../data/hnph_shuffle_188.txt',  # 共享标签文件
        return_filenames=True
    )
    val_dataset = test_dataset = datasets.MRIDataset(
        root_dir_fa=osp.join(root, 'hnph'),
        labels_file='../../../data/hnph_shuffle_188.txt',  # 共享标签文件
        return_filenames=True
    )
    class_names = ['healthy', 'pd']
    num_classes = len(class_names)

    return train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names

import itertools
def plot_confusion_matrix(cm):

    plt.figure()
    # 可视化混淆矩阵
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=14)
    plt.colorbar()
    indices = range(cm.shape[0])
    labels = [0,1]
    plt.xticks(indices, labels, fontsize=14)
    plt.yticks(indices, labels, fontsize=14)

    # 在混淆矩阵中添加文本标注
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
        horizontalalignment="center", fontsize=16)

    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.savefig('mask_p2h_cm.png', dpi=600)
    plt.show()

def test(test_loader, model, args, device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    test_trues = None
    test_probs = None

    with torch.no_grad():
        end = time.time()
        for i, (images, target, filename, _) in enumerate(test_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = F.cross_entropy(output, target)

            logit = torch.nn.functional.softmax(output, dim=1)
            test_probs = logit.cpu().numpy() if test_probs is None \
                else np.concatenate([test_probs, logit.cpu().numpy()])

            test_trues = target.cpu().numpy() if test_trues is None \
                else np.concatenate([test_trues, target.cpu().numpy()])

            # measure accuracy and record loss
            acc1, = accuracy(output, target, topk=(1,))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        if confmat:
            print(confmat.format(args.class_names))

        test_preds = np.argmax(test_probs, 1)

        macro_roc_auc_ovr = roc_auc_score(
            test_trues,
            np.array(test_probs)[:,1],
            multi_class="ovr",
            average="macro",
        )
        print(f"Macro-averaged One-vs-Rest ROC AUC score:{macro_roc_auc_ovr:.4f}")

        fpr, tpr, thresholds = roc_curve(test_trues, np.array(test_probs)[:,1], pos_label=1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % macro_roc_auc_ovr)
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig('mask_p2h_roc.png', dpi=600)
        # plt.show()

        print(classification_report(test_trues, test_preds, digits=4))

        conf_matrix = confusion_matrix(test_trues, test_preds)
        print(conf_matrix)
        plot_confusion_matrix(conf_matrix)
    return top1.avg

def main(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Data loading code
    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        get_dataset(args.root)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = get_model(args.arch, pretrain=not args.scratch)
    print(backbone)
    pool_layer = nn.Identity() if args.no_pool else None
    print(args.no_pool, pool_layer)
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=not args.scratch).to(device)
    classifier_feature_dim = classifier.features_dim

    # evaluate on test set
    net.load_state_dict(torch.load('path-to-your-checkpoint/checkpoint_epoch_2000.pth')['model_state_dict'])

    acc1 = test(test_loader, classifier, args, device)
    print("test_acc1 = {:3.1f}".format(acc1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CDAN+MCC with SDAT for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('--root', type=str, default='../../../data',
                        help='root path of dataset')
    parser.add_argument('-a', '--arch', default='ResNet18_3D')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--gpu', type=str, default="1", help="GPU ID")
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--scratch', action='store_true',
                        help='whether train from scratch.')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    main(args)