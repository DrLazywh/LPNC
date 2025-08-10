import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import os.path as op
from datasets import build_dataloader
from processor.processor import do_inference1
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
import argparse
from utils.iotools import load_train_configs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TranTextReID Text")
    sub = 'PPL_logs/CUHK-PEDES/20250106_165227_PPL_supid+cotrl+cid'
    parser.add_argument("--config_file", default=f'{sub}/configs.yaml')
    args = parser.parse_args()
    args = load_train_configs(args.config_file)
    args.training = False
    logger = setup_logger('PPL', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"
    args.output_dir = sub
    test_img_loader, test_txt_loader,refer, num_classes = build_dataloader(args)
    asss = ['best.pth']
    for i in range(len(asss)):
        if os.path.exists(op.join(args.output_dir, asss[i])):
            model = build_model(args,num_classes)
            checkpointer = Checkpointer(model)
            checkpointer.load(f=op.join(args.output_dir, asss[i]))
            model = model.cuda()
            do_inference1(model, test_img_loader, test_txt_loader, refer,args)