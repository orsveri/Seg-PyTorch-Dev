from __future__ import print_function
import os
import cv2
import json
import time
import torch
import random
import shutil
import datetime
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from tools.model_utils import model_dict, save_checkpoint
from tools.data_utils import generator, generator_test, generator_crops, generator_crops_test


# ------------------------------------------------------------------------------------------------------------------ #
# Входные параметры
# ------------------------------------------------------------------------------------------------------------------ #

# Номер разбиения данных
cv_n = None
model_name = 'resnet34'
data_name = 'aviation'
logdir = '/lockdown_neuro/home/user/orlova/Projects/dpf_research_DATA/1_class_feat_extr/1_CLASSIFICATION_Pytorch_simple/logs/TEST{}_resnet34_fill_outframe_test{}'.format(data_name, '-cv{}'.format(cv_n) if cv_n else '')
epochs = 30
batch_size = 8 # 64
input_shape = (256, 256) # Это вместе с паддингом
input_padding = (0, 0) # (tb, lr), tb - top-bottom, lr - left-right. Это значение будет добавлено к КАЖДОЙ стороне.
data_rootdir = '/lockdown_neuro/home/user/data/aviation/classifier' # '/home/user/data/ppr_copy' # '/mnt/share/data/datasets/ppr' # '/lockdown_neuro/home/user/data/face_mask/dataset' # '/home/user/data/face_mask/dataset'
data_t = '/lockdown_neuro/home/user/orlova/Projects/dpf_research_DATA/1_class_feat_extr/1_CLASSIFICATION_Pytorch_simple/data/{}/data_train{}.csv'.format(data_name, '_cv{}'.format(cv_n) if cv_n else '')
data_v = '/lockdown_neuro/home/user/orlova/Projects/dpf_research_DATA/1_class_feat_extr/1_CLASSIFICATION_Pytorch_simple/data/{}/data_test{}.csv'.format(data_name, '_cv{}'.format(cv_n) if cv_n else '')
class_id_json = '/lockdown_neuro/home/user/orlova/Projects/dpf_research_DATA/1_class_feat_extr/1_CLASSIFICATION_Pytorch_simple/data/{}/class_ids.json'.format(data_name)
max_steps_per_epoch = 3 #5470
gpus = "0"
print_every = 1
test_every = 1
do_aug = True
generator_with_crop = False
resize_pad = False
fill_outframe = True
fill_outframe_test = True

# Путь к предобученной модели <name>
back_path_root = '/lockdown_neuro/home/user/orlova/Projects/dpf_research_DATA/1_class_feat_extr/1_CLASSIFICATION_Pytorch_simple'
back_path = os.path.join(back_path_root, model_dict[model_name]['imagenet_weights']) # 'pretrained_weights/resnet34-333f7ec4.pth' # 'pretrained_weights/resnet50-19c8e357.pth' # None
full_weights_path = None # 'logs/FGVC-Aircraft-01_03/model.pth.tar-135'

# ------------------------------------------------------------------------------------------------------------------ #
# Подготовим параметры для конфига и данные
# ------------------------------------------------------------------------------------------------------------------ #

train_config = {}

# Убедимся, что есть куда сохранять логи
try:
    os.mkdir(os.path.join(logdir))
except FileExistsError:
    if any(file.startswith('events.out.') for file in os.listdir(os.path.join(logdir))):
        raise Exception(
            'Нельзя записывать логи новой обучающей сессии в папку с другими логами! Логи не должны смешиваться!!\nСмените logdir в конфиге')

train_config["logdir"] = logdir

with open(class_id_json, "r") as f:
    class_id = json.load(f)
nb_classes = len(class_id)

train_config["class_list"] = [k for k, v in sorted(class_id.items(), key = lambda x: x[1])]
train_config["class_ids_json"] = class_id_json
train_config["input_shape"] = [input_shape[0], input_shape[1], 3]
train_config["input_shape_padding"] = list(input_padding)
train_config["num_classes"] = nb_classes
train_config["data_train"] = data_t
train_config["data_valtest"] = data_v
train_config["with_augment"] = do_aug
train_config["resize_pad"] = resize_pad
train_config["fill_outframe"] = fill_outframe
train_config["fill_outframe_test"] = fill_outframe_test

L_train = pd.read_csv(data_t).shape[0]
spe = L_train//batch_size
if spe > max_steps_per_epoch: spe = max_steps_per_epoch
L_test  = pd.read_csv(data_v).shape[0]
spe_test = L_test//batch_size

if generator_with_crop:
    gen_t = generator_crops(batch_size=batch_size,
                            input_shape=input_shape,
                            data=data_t,
                            classnames_to_int=class_id,
                            do_augment=do_aug,
                            data_rootdir=data_rootdir,
                            input_shape_padding=input_padding)
    gen_v = generator_crops_test(batch_size=batch_size, input_shape=input_shape, data=data_v,
                                 classnames_to_int=class_id, n_samples=None,  # batch_size*spe_test,
                                 data_rootdir=data_rootdir, input_shape_padding=input_padding)
else:
    gen_t = generator(batch_size=batch_size,
                      input_shape=input_shape,
                      data=data_t,
                      classnames_to_int=class_id,
                      do_augment=do_aug,
                      data_rootdir=data_rootdir,
                      resize_pad=resize_pad,
                      fill_outframe=fill_outframe)
    gen_v = generator_test(batch_size=batch_size,
                           input_shape=input_shape,
                           data=data_v,
                           classnames_to_int=class_id,
                           n_samples=None, #batch_size*spe_test,
                           data_rootdir=data_rootdir,
                           resize_pad=resize_pad,
                           fill_outframe=fill_outframe_test)

# Сохраним конфиг
with open(os.path.join(logdir, "cfg.json"), 'w', encoding="utf-8") as f:
    json.dump(train_config, f, indent=4)

# ------------------------------------------------------------------------------------------------------------------ #
# Подготовим модель
# ------------------------------------------------------------------------------------------------------------------ #

# Загружаем модель
print('Building model...')
model = model_dict[model_name]['builder'](num_classes=1000) # nb_classes
if back_path:
    state_dict = torch.load(back_path)
    model.load_state_dict(state_dict, strict = False)

# Sigmoid есть в loss!
# model.fc = torch.nn.Sequential(torch.nn.Linear(in_features=model_dict[model_name]['out_shape'], out_features=nb_classes), torch.nn.Sigmoid())
model.fc = torch.nn.Sequential(torch.nn.Linear(in_features=model_dict[model_name]['out_shape'],
                                               out_features=nb_classes))

if full_weights_path:
    model.load_state_dict(torch.load(full_weights_path))

model.train()
if gpus:
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0., betas=(0.9, 0.999))
criterion = nn.CrossEntropyLoss().cuda()
writer = SummaryWriter(log_dir=logdir)

for epoch in range(epochs):
    model.train()

    start_time = time.time()
    sum_batch_time = 0
    sum_loss = 0
    sum_accuracy = 0

    for step in range(spe):
        imgs, lbls = next(gen_t)
        imgs = imgs.cuda()
        lbls = lbls.cuda()

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        correct_batch = (outputs.max(1)[1] == lbls).float().sum()

        # estimate remaining time
        delta_seconds = time.time() - start_time
        eta_seconds = delta_seconds / (step+1) * (spe-step-1)
        delta_str = str(datetime.timedelta(seconds=int(delta_seconds)))
        eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))

        sum_batch_time += delta_seconds
        sum_loss += loss.item()
        sum_accuracy += correct_batch

        print('Epoch: [{0}/{1}][{2}/{3}]\tTime passed {batch_time:.2f} (remains {eta})\tLoss {loss:.4f})\tAcc {acc:.2f}'.format(
            epoch+1, epochs, step+1, spe,
            batch_time=delta_seconds,
            eta=eta_str,
            loss=loss.item(),
            acc=correct_batch/batch_size))

    # Сохраним логи в тензорборд и чекпоинт
    writer.add_scalar('Train/Time', sum_batch_time/(spe*batch_size), epoch)
    writer.add_scalar('Train/Loss', sum_loss/(spe*batch_size), epoch)
    writer.add_scalar('Train/Acc', sum_accuracy/(spe*batch_size), epoch)
    writer.add_scalar('Train/Lr', optimizer.param_groups[0]['lr'], epoch)
    save_checkpoint(state=model.state_dict(), save_dir=logdir, epoch=epoch, is_best=False)

    # Testing
    if epoch % test_every == 0:
        model.eval()

        acc_test = 0
        n_samples = batch_size*spe_test
        batch_time_test = 0
        loss_test = 0

        for step in range(spe_test):
            imgs, lbls = next(gen_v)
            imgs = imgs.cuda()
            lbls = lbls.cuda()

            #optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            #optimizer.step()

            correct_batch = (outputs.max(1)[1] == lbls).float().sum()

            # estimate remaining time
            delta_seconds = time.time() - start_time
            eta_seconds = delta_seconds / (step+1) * (spe_test - step-1)
            delta_str = str(datetime.timedelta(seconds=int(delta_seconds)))
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))

            print('VAL Epoch, step: [{0}/{1}]\tTime passed {batch_time} (remains {eta})\tLoss {loss:.4f})\tAcc {acc:.2f}'.format(
                step+1, spe_test, batch_time=delta_seconds, eta=eta_str, loss=loss.item(), acc=correct_batch/batch_size))

            loss_test += loss.item()
            batch_time_test += delta_seconds
            acc_test += correct_batch

        # Сохраним логи в тензорборд
        writer.add_scalar('Val/Time', batch_time_test / n_samples, global_step=epoch)
        writer.add_scalar('Val/Loss', loss_test / n_samples, global_step=epoch)
        writer.add_scalar('Val/Acc', acc_test / n_samples, global_step=epoch)

        print('Testing epoch is done!')



