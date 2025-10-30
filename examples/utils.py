import sys
import os.path as osp
import time
from copy import deepcopy
from functools import partial
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from timm.models import register_model
from torch.utils.data import ConcatDataset
import wandb
import wilds

from dalib.modules.mae import create_mae

sys.path.append('../')
import common.vision.datasets as datasets
import common.vision.models as models
from common.vision.transforms import ResizeImage
from common.utils.metric import accuracy, ConfusionMatrix
from common.utils.meter import AverageMeter, ProgressMeter


def get_model_names():
    model_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()
    return model_names


@register_model
def mae_base_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12,
                        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        **kwargs)
    model = create_mae('mae_base_patch16_224', pretrained=pretrained, **model_kwargs)

    return model


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


def convert_from_wilds_dataset(wild_dataset):
    class Dataset:
        def __init__(self):
            self.dataset = wild_dataset

        def __getitem__(self, idx):
            x, y, metadata = self.dataset[idx]
            return x, y

        def __len__(self):
            return len(self.dataset)

    return Dataset()


def convert_dataset(dataset):
    """
    Converts a source dataset which returns (img, label) pairs into one that returns (d_label, img, label) triplets.
    """

    class DatasetWrapper:

        def __init__(self):
            self.dataset = dataset

        def __getitem__(self, index):
            d_label = torch.ones(1)
            return d_label, self.dataset[index]

        def __len__(self):
            return len(self.dataset)

    return DatasetWrapper()


def get_dataset_names():
    return sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    ) + wilds.supported_datasets + ['Digits']


def get_dataset(dataset_name, root, source, target, train_source_transform, val_transform, train_target_transform=None):
    if train_target_transform is None:
        train_target_transform = train_source_transform
    if dataset_name == "Digits":
        train_source_dataset = datasets.__dict__[source[0]](osp.join(root, source[0]), download=True,
                                                            transform=train_source_transform)
        train_target_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), download=True,
                                                            transform=train_target_transform)
        val_dataset = test_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), split='test',
                                                                  download=True, transform=val_transform)
        class_names = datasets.MNIST.get_classes()
        num_classes = len(class_names)
    elif dataset_name in datasets.__dict__:
        # load datasets from common.vision.datasets
        dataset = datasets.__dict__[dataset_name]

        def concat_dataset(tasks, **kwargs):
            return ConcatDataset([dataset(task=task, **kwargs) for task in tasks])

        train_source_dataset = concat_dataset(root=root, tasks=source, download=True, transform=train_source_transform)
        train_target_dataset = concat_dataset(root=root, tasks=target, download=True, transform=train_target_transform)
        val_dataset = concat_dataset(root=root, tasks=target, download=True, transform=val_transform)
        if dataset_name == 'DomainNet':
            test_dataset = concat_dataset(root=root, tasks=target, split='test', download=True, transform=val_transform)
        else:
            test_dataset = val_dataset
        class_names = train_source_dataset.datasets[0].classes
        num_classes = len(class_names)
    elif dataset_name == 'pd':
        train_source_dataset = datasets.MRIDataset(
            root_dir_fa=osp.join(root, 'hnph'),
            labels_file='../../../data/hnph_shuffle.txt',  # 共享标签文件
            return_filenames=False
        )
        train_target_dataset =  datasets.MRIDataset(
            root_dir_fa=osp.join(root, 'ppmi1'),
            labels_file='../../../data/ppmi1_shuffle.txt',  # 共享标签文件
            return_filenames=True
        )
        val_dataset = test_dataset = datasets.MRIDataset(
            root_dir_fa=osp.join(root, 'ppmi1'),
            labels_file='../../../data/ppmi1_shuffle.txt',  # 共享标签文件
            return_filenames=True
        )
        # train_source_dataset = datasets.MRIDataset(
        #     root_dir_fa=osp.join(root, 'ppmi1'),
        #     labels_file='../../../data/ppmi1_shuffle.txt',  # 共享标签文件
        #     return_filenames=False
        # )
        # train_target_dataset =  datasets.MRIDataset(
        #     root_dir_fa=osp.join(root, 'hnph'),
        #     labels_file='../../../data/hnph_shuffle_188.txt',  # 共享标签文件
        #     return_filenames=True
        # )
        # val_dataset = test_dataset = datasets.MRIDataset(
        #     root_dir_fa=osp.join(root, 'hnph'),
        #     labels_file='../../../data/hnph_shuffle_188.txt',  # 共享标签文件
        #     return_filenames=True
        # )
        class_names = ['healthy', 'pd']
        num_classes = len(class_names)
    else:
        # load datasets from wilds
        dataset = wilds.get_dataset(dataset_name, root_dir=root, download=True)
        num_classes = dataset.n_classes
        class_names = None
        train_source_dataset = convert_from_wilds_dataset(dataset.get_subset('train', transform=train_source_transform))
        train_target_dataset = convert_from_wilds_dataset(dataset.get_subset('test', transform=train_target_transform))
        val_dataset = test_dataset = convert_from_wilds_dataset(dataset.get_subset('test', transform=val_transform))
    return train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names


class EntropyLoss(nn.Module):
    """
    This class contains the entropy function implemented.
    """
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x, applySoftMax=True):
        #Assuming x : [BatchSize, ]

        if applySoftMax:
            entropy = torch.nn.functional.softmax(x, dim=1)*torch.nn.functional.log_softmax(x, dim=1)
        else:
            entropy = x * torch.log2(x)
        entropy = -1*entropy.sum(dim=1)
        return entropy


def entropy(probs):
    log_probs = torch.log(probs)
    entropy = torch.sum(-probs * log_probs, dim=1)
    return entropy

# def validate(val_loader, model, args, device) -> float:
def validate(val_loader, model, args, device):

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Val: ')


    # ### Bayesian Active Learning by Disagreement (BALD) extension
    # all_score = None
    # all_entropy = None
    # targets = None
    # mc_dropout_iterations = 25
    # model.train()
    #
    # for j in range(mc_dropout_iterations):
    #     scores = None
    #
    #     for i, (data_x, data_y, _) in enumerate(val_loader):
    #         data_x = data_x.cuda(non_blocking=True)
    #         data_y = data_y.cuda(non_blocking=True)
    #
    #         with torch.no_grad():
    #             output, _ = model(data_x)
    #             output = torch.nn.functional.softmax(output, dim=1)
    #
    #         scores = output if scores is None else torch.cat([scores, output])
    #         # targets = data_y.cpu().numpy() if targets is None \
    #         #     else np.concatenate([targets, data_y.cpu().numpy()])
    #
    #     # print('\n MC dropout sample: ', j+1)
    #     # print(targets)
    #
    #     all_score = scores if all_score is None else all_score + scores
    #     all_entropy = entropy(scores) if all_entropy is None else all_entropy + entropy(scores)
    #
    # avg_score = all_score / mc_dropout_iterations
    # entropy_avg_score = entropy(avg_score)
    #
    # average_entropy = all_entropy / mc_dropout_iterations
    #
    # scores = entropy_avg_score - average_entropy
    #
    # samples_indices = scores.argsort(descending=True)[:5]
    # probs_max, predict = torch.max(avg_score, 1)


    ## deep bayesian active learning where uncertainty is measured by maximizing entropy of predictions
    model.train()
    for m in model.modules():
        # print("True")
        if isinstance(m, torch.nn.BatchNorm3d):
            m.eval()
    DROPOUT_ITERATIONS = 25
    entropy_loss = EntropyLoss()
    u_scores = None
    probs = None
    targets = None

    for i, (images, data_y, _, _) in enumerate(val_loader):
        images = images.to(device)
        data_y = data_y.to(device)
        z_op = np.zeros((images.shape[0], 2), dtype=float)

        targets = data_y.cpu().numpy() if targets is None \
            else np.concatenate([targets, data_y.cpu().numpy()])

        for j in range(DROPOUT_ITERATIONS):
            with torch.no_grad():
                temp_op, _ = model(images)
                # Till here z_op represents logits of p(y|x).
                # So to get probabilities
                temp_op = torch.nn.functional.softmax(temp_op, dim=1)
                z_op = np.add(z_op, temp_op.cpu().numpy())

        z_op /= DROPOUT_ITERATIONS
        z_op = torch.from_numpy(z_op).cuda(non_blocking=True)
        probs = z_op if probs is None else torch.cat([z_op, probs])

        entropy_z_op = entropy_loss(z_op, applySoftMax=False)

        # Now entropy_z_op = Sum over all classes{ -p(y=c|x) log p(y=c|x)}
        u_scores = entropy_z_op if u_scores is None else torch.cat([u_scores, entropy_z_op])

    samples_indices = u_scores.argsort(descending=False)[:5]
    # # print(samples_indices)

    probs_max, predict = torch.max(probs, 1)
    # samples_indices = torch.where(probs_max > 0.99)[0]
    # print(probs_max, predict)

    # indices = np.where(targets == predict.cpu().numpy())[0]
    # idx = probs_max[indices].argsort(descending=True)[:5]
    # samples_indices = indices[idx.cpu().numpy()]
    # print(targets.shape, predict.shape, probs_max[indices].shape, np.where(targets==predict.cpu().numpy()))

    print(samples_indices)

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
        filenames_wrong = []
        for i, (images, target, filename, _) in enumerate(val_loader):
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

            probs_max, pred = torch.max(logit, 1)
            ind = torch.where(pred != target)[0].cpu().numpy()
            for i in range(len(ind)):
                filenames_wrong.append(filename[ind[i]])
            # measure accuracy and record loss
            acc1, = accuracy(output, target, topk=(1,))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # # 获取当前时间并格式化
        # current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # # 定义基础文件名和扩展名
        # base_filename = "wrong_filename"
        # file_extension = ".txt"
        # # 拼接完整文件名
        # filename = f"{base_filename}_{current_time}{file_extension}"

        # probs_max = np.max(test_probs, 1)
        # predict = np.argmax(test_probs, 1)
        # indices = np.where(test_trues == predict)[0]
        # idx = np.argsort(-probs_max[indices])[:5]
        # samples_indices = indices[idx]

        with open('wrong_filename_h2p.txt', 'a') as f:
            f.write(f"{filenames_wrong}\n")

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        if confmat:
            print(confmat.format(args.class_names))

        # test_preds = np.argmax(test_probs, 1)
        #
        # # sklearn_accuracy = accuracy_score(test_trues, test_preds)
        # # sklearn_precision = precision_score(test_trues, test_preds, average='macro')
        # # sklearn_recall = recall_score(test_trues, test_preds, average='macro')
        # # sklearn_f1 = f1_score(test_trues, test_preds, average='macro')
        # macro_roc_auc_ovr = roc_auc_score(
        #     test_trues,
        #     np.array(test_probs)[:,1],
        #     multi_class="ovr",
        #     average="macro",
        # )
        # print(f"Macro-averaged One-vs-Rest ROC AUC score:{macro_roc_auc_ovr:.4f}")
        #
        # # fpr, tpr, thresholds = roc_curve(test_trues, np.array(test_probs)[:,1], pos_label=1)
        # # auc1 = auc(fpr, tpr)
        # # print("auc:{:.4f}".format(auc1))
        #
        # print(classification_report(test_trues, test_preds, digits=4))
        #
        # conf_matrix = confusion_matrix(test_trues, test_preds)
        # print(conf_matrix)
        # # plot_confusion_matrix(conf_matrix)
        # # print("[sklearn_metrics] accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(sklearn_accuracy,
        # #                                                                                           sklearn_precision,
        # #                                                                                           sklearn_recall,
        # #                                                                                           sklearn_f1))

    # print(targets.shape, predict.shape, np.where(targets==predict.cpu().numpy()))

    return top1.avg, samples_indices, predict.cpu().numpy()
    # return top1.avg, samples_indices, predict


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

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        if confmat:
            print(confmat.format(args.class_names))

        test_preds = np.argmax(test_probs, 1)

        # sklearn_accuracy = accuracy_score(test_trues, test_preds)
        # sklearn_precision = precision_score(test_trues, test_preds, average='macro')
        # sklearn_recall = recall_score(test_trues, test_preds, average='macro')
        # sklearn_f1 = f1_score(test_trues, test_preds, average='macro')
        macro_roc_auc_ovr = roc_auc_score(
            test_trues,
            np.array(test_probs)[:,1],
            multi_class="ovr",
            average="macro",
        )
        print(f"Macro-averaged One-vs-Rest ROC AUC score:{macro_roc_auc_ovr:.4f}")

        # fpr, tpr, thresholds = roc_curve(test_trues, np.array(test_probs)[:,1], pos_label=1)
        # auc1 = auc(fpr, tpr)
        # print("auc:{:.4f}".format(auc1))

        print(classification_report(test_trues, test_preds, digits=4))

        conf_matrix = confusion_matrix(test_trues, test_preds)
        print(conf_matrix)
        # plot_confusion_matrix(conf_matrix)
        # print("[sklearn_metrics] accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(sklearn_accuracy,
        #                                                                                           sklearn_precision,
        #                                                                                           sklearn_recall,
        #                                                                                           sklearn_f1))
    return top1.avg


def get_train_transform(resizing='default', random_horizontal_flip=True, random_color_jitter=False,
                        resize_size=224, norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    """
    resizing mode:
        - default: resize the image to 256 and take a random resized crop of size 224;
        - cen.crop: resize the image to 256 and take the center crop of size 224;
        - res: resize the image to 224;
    """
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224)
        ])
    elif resizing == 'cen.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224)
        ])
    elif resizing == 'ran.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomCrop(224)
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if random_color_jitter:
        transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])
    return T.Compose(transforms)


def get_val_transform(resizing='default', resize_size=224,
                      norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    """
    resizing mode:
        - default: resize the image to 256 and take the center crop of size 224;
        – res.: resize the image to 224
    """
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])


def pretrain(train_source_iter, model, optimizer, lr_scheduler, epoch, args, device):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, f_s = model(x_s)

        cls_loss = F.cross_entropy(y_s, labels_s)
        loss = cls_loss

        cls_acc = accuracy(y_s, labels_s)[0]
        if args.log_results:
            wandb.log({'iteration':epoch*args.iters_per_epoch + i, 'loss':loss})

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
