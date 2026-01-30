# -*- coding: utf-8 -*-
# @Description: Main process of network testing.
# @Author: Zhe Zhang (doublez@stu.pku.edu.cn)
# @Affiliation: Peking University (PKU)
# @LastEditDate: 2023-09-07

import os, time, sys, gc, cv2, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils.eval_depth_metrics import compute_abs_rel_error, compute_thresh_metrics


from datasets.data_io import *
from datasets.dtu import DTUDataset
from datasets.tnt import TNTDataset
from datasets.blendedmvs import BlendedMVSDataset


from models.geomvsnet import GeoMVSNet
from models.utils import *
from models.utils.opts import get_opts


cudnn.benchmark = True

args = get_opts()


def test():
    total_time = 0
    total_abs_rel = 0.0
    total_thresh_1 = 0.0
    total_thresh_2 = 0.0
    total_thresh_3 = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            sample_cuda = tocuda(sample)
            start_time = time.time()

            # @Note GeoMVSNet main
            outputs = model(
                sample_cuda["imgs"], 
                sample_cuda["depths"],   # <-- Added depth input
                sample_cuda["proj_matrices"], sample_cuda["intrinsics_matrices"], 
                sample_cuda["depth_values"], 
                sample["filename"]
            )

            end_time = time.time()
            total_time += end_time - start_time
            outputs = tensor2numpy(outputs)
            # Depth evaluation
            #depth_preds = outputs["depth"]        # [B, H, W]
            #depth_gts = sample["depth_gt"]        # [B, H, W] (assumed from dataset)
            #masks = sample["mask"]                # [B, H, W]

            depth_preds = torch.from_numpy(outputs["depth"]).float().cuda()  # convert model output to tensor
            if "depth_gt" in sample and "mask" in sample:
                depth_gts = sample["depth_gt"]                              # already float on CUDA
                masks = sample["mask"]                                      # bool on CUDA


                for pred, gt, mask in zip(depth_preds, depth_gts, masks):
                    pred = torch.from_numpy(pred).float().cuda()
                    gt = gt.cuda()
                    mask = mask.cuda()

    	
                    abs_rel = compute_abs_rel_error(pred, gt, mask)
                    thresh1, thresh2, thresh3 = compute_thresh_metrics(pred, gt, mask)

                    total_abs_rel += abs_rel
                    total_thresh_1 += thresh1
                    total_thresh_2 += thresh2
                    total_thresh_3 += thresh3
                    num_samples += 1
            else:
                logger.warning("No ground-truth depth/mask found — skipping metric evaluation for this batch.")

            del sample_cuda

            filenames = sample["filename"]
            cams = sample["proj_matrices"]["stage{}".format(args.levels)].numpy()
            imgs = sample["imgs"]
            logger.info('Iter {}/{}, Time:{:.3f} Res:{}'.format(batch_idx, len(TestImgLoader), end_time - start_time, imgs[0].shape))


            for filename, cam, img, depth_est, photometric_confidence in zip(filenames, cams, imgs, outputs["depth"], outputs["photometric_confidence"]):
                img = img[0].numpy()    # ref view
                cam = cam[0]            # ref cam
                print("Filenames from dataset:", filenames)
                
                #create scene dir
                scene_id = filename.split('/')[0]  # "58c4bf..."
                scene_dir = os.path.join(args.outdir, scene_id, "predictions")
                os.makedirs(scene_dir, exist_ok=True)
                
                base_name = filename.split('/')[-1]

                
                depth_filename = os.path.join(scene_dir, base_name + "_depth_est.pfm")

                #scene_dir = os.path.join(args.outdir, os.path.dirname(filename))
                #os.makedirs(scene_dir, exist_ok=True)
                print("Saving outputs to:", args.outdir)
                print("Saving", filename)
                    

                base_name = os.path.basename(filename)
                depth_filename = os.path.join(scene_dir, base_name + "_depth_est.pfm")
                confidence_filename = os.path.join(scene_dir, base_name + "_confidence.pfm") 
                cam_filename = os.path.join(scene_dir, base_name + "_cam.txt")
                img_filename = os.path.join(scene_dir, base_name + "_ref.jpg")
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                #if args.which_dataset == 'dtu':
                os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)
                
                # save depth maps
                save_pfm(depth_filename, depth_est)

                # save confidence maps
                confidence_list = [outputs['stage{}'.format(i)]['photometric_confidence'].squeeze(0) for i in range(1,5)]
                photometric_confidence = confidence_list[-1]
                if not args.save_conf_all_stages:
                    save_pfm(confidence_filename, photometric_confidence) 
                else:
                    for stage_idx, photometric_confidence in enumerate(confidence_list):
                        if stage_idx != args.levels - 1:
                            confidence_filename = os.path.join(args.outdir, filename.format('confidence', "_stage"+str(stage_idx)+'.pfm'))
                        else:
                            confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                        save_pfm(confidence_filename, photometric_confidence) 

                # save cams, img
                #if args.which_dataset == 'dtu':
                write_cam(cam_filename, cam)
                img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_filename, img_bgr)

    torch.cuda.empty_cache()
    gc.collect()
    if num_samples > 0:
    	avg_abs_rel = total_abs_rel / num_samples
    	avg_thresh_1 = total_thresh_1 / num_samples
    	avg_thresh_2 = total_thresh_2 / num_samples
    	avg_thresh_3 = total_thresh_3 / num_samples

    	logger.info(f"[Evaluation] Samples: {num_samples}")
    	logger.info(f"[Evaluation] AbsRel: {avg_abs_rel:.4f}")
    	logger.info(f"[Evaluation] Thresh d<1.25: {avg_thresh_1:.4f}, d<1.25^2: {avg_thresh_2:.4f}, d<1.25^3: {avg_thresh_3:.4f}")

    return total_time, len(TestImgLoader)


def initLogger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    curTime = time.strftime('%Y%m%d-%H%M', time.localtime(time.time()))

    if args.which_dataset == 'tnt':
        logfile = os.path.join(args.logdir, 'TNT-test-' + curTime + '.log')
    else:
        logfile = os.path.join(args.logdir, 'test-' + curTime + '.log')
    
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    if not args.nolog:
        fileHandler = logging.FileHandler(logfile, mode='a')
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    logger.info("Logger initialized.")
    logger.info("Writing logs to file: {}".format(logfile))
    logger.info("Current time: {}".format(curTime))

    settings_str = "All settings:\n"
    for k,v in vars(args).items(): 
        settings_str += '{0}: {1}\n'.format(k,v)
    logger.info(settings_str)

    return logger


if __name__ == '__main__':
    logger = initLogger()

    # dataset, dataloader
    if args.which_dataset == 'dtu':
        test_dataset = DTUDataset(args.testpath, args.testlist, "test", args.n_views, max_wh=(1600, 1200))
    elif args.which_dataset == 'tnt':
        test_dataset = TNTDataset(args.testpath, args.testlist, split=args.split, n_views=args.n_views, img_wh=(-1, 1024), cam_mode=args.cam_mode, img_mode=args.img_mode)
    elif args.which_dataset == 'blendedmvs':
        test_dataset = BlendedMVSDataset(
        root_dir=args.testpath,
        list_file=args.testlist,
        split="val",
        n_views=args.n_views,
        img_wh=(768, 576),     # or your training resolution
        robust_train=False,
        augment=False
    )

    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    # @Note GeoMVSNet model
    model = GeoMVSNet(
        levels=args.levels, 
        hypo_plane_num_stages=[int(n) for n in args.hypo_plane_num_stages.split(",")], 
        depth_interal_ratio_stages=[float(ir) for ir in args.depth_interal_ratio_stages.split(",")],
        feat_base_channel=args.feat_base_channel, 
        reg_base_channel=args.reg_base_channel,
        group_cor_dim_stages=[int(n) for n in args.group_cor_dim_stages.split(",")],
    )
    
    logger.info("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['model'], strict=False)

    model.cuda()
    model.eval()

    test()
