# pytorch-lightning
from multiprocessing.dummy import current_process
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger

from config import get_opts

from SC_Depth import SC_Depth
from SC_DepthV2 import SC_DepthV2

import os
import torch

if __name__ == '__main__':
    hparams = get_opts()
    
    current_path = os.getcwd() + '/kitti_scv1/version_0/epoch=99-val_loss=0.1411.ckpt'

    if hparams.model_version == 'v1':
        system = SC_Depth(hparams, current_path)
    elif hparams.model_version == 'v2':
        system = SC_DepthV2(hparams)

    logger = TestTubeLogger(
        save_dir="ckpts",
        name=hparams.exp_name,
        debug=False,
        create_git_tag=False
    )

    ckpt_dir = 'ckpts/{}/version_{}'.format(hparams.exp_name, logger.experiment.version)

    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                          filename='{epoch}-{val_loss:.4f}',
                                          monitor='val_loss',
                                          mode='min',
                                          save_last=True,
                                          save_weights_only=True,
                                          save_top_k=3)

    if hparams.ckpt_path is not None:
        print('load pre-trained model from {}'.format(hparams.ckpt_path))
        system = system.load_from_checkpoint(hparams.ckpt_path)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                    #   limit_train_batches=hparams.epoch_size,
                      num_sanity_val_steps=5,
                      min_epochs = 30,
                      callbacks=[checkpoint_callback],
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=1,
                      accelerator='gpu',
                      devices=[6, 7],
                      strategy="ddp",
                      benchmark=True)

    trainer.fit(system)
