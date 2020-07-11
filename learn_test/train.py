# -*- coding: utf-8 -*-
import os
import sys
import time
import logging
import yaml


import torch
import torch.nn as nn
import torch.optim as optimal

from DL.basic import PredictMethod
from scripts import loader

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)
# torch.backends.cudnn.benchmark = True


def main():
    with open('../config/VanillaLSTM_params.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config['Model']['Architecture'])
    algorithm = PredictMethod(config['Model']).method
    train_load = loader.data_loader(config['Datasets'])

    logger.info('{} iterations per epoch', format(config['Datasets']['Batch_size']))

    restore_path = None
    checkpoint_name = config['Train']['Check_point_name']
    if config['Train']['Model_dir'] is not None:
        restore_path = config['Train']['Model_dir']
    elif config['Train']['Restore_from_checkpoint']:
        restore_path = os.path.join(config['Model']['Architecture'], '%s_model.pt' % checkpoint_name)

    # 若checkpoint文件存在且restore为真则加载checkpoint
    if restore_path is not None and os.path.isfile(restore_path) and config['Train']['Restore_from_checkpoint']:
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path)

    # 否则重新开始checkpoint
    else:
        checkpoint = {}

    # 检查epoch合法性
    epoch = config['Train']['Epoch']
    if epoch > len(train_load)/config['Datasets']['Batch_size']:
        epoch = int(len(train_load)/config['Datasets']['Batch_size'])
        logger.warn('epoch too large, reset to legal_max: ', epoch)
    t = 0
    while t < epoch:
        for batch in train_load:
            torch.cuda.synchronize()
            start = time.time()
            algorithm(batch)
            '''
            backward
            '''
            torch.save(checkpoint, config['Train']['Model_dir'])
            logger.info('Done')
        t += 1


if __name__ == '__main__':
    main()
