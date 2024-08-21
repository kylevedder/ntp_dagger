# Long-term trajectory estimation with NTP.
# Unofficial implementation of NTP - CVPR'2022 (https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Neural_Prior_for_Trajectory_Estimation_CVPR_2022_paper.pdf)

import os, glob
import argparse
import logging
import csv
import numpy as np
import torch
import sys
import pytorch3d.loss as p3dloss
from pathlib import Path
from bucketed_scene_flow_eval.utils import load_feather, save_feather
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils.general_utils import *
from utils.ntp_utils import *
from bucketed_scene_flow_eval.datasets import construct_dataset
from bucketed_scene_flow_eval.datastructures import (
    TimeSyncedSceneFlowFrame,
    O3DVisualizer,
    PointCloud,
)
from dataclasses import dataclass
import tqdm

logger = logging.getLogger(__name__)


def load_sequence(
    sequence_data_folder: Path,
    flow_folder: Path,
    sequence_length: int,
    sequence_id: str,
) -> list[TimeSyncedSceneFlowFrame]:
    dataset = construct_dataset(
        name="Argoverse2NonCausalSceneFlow",
        args=dict(
            root_dir=sequence_data_folder,
            subsequence_length=sequence_length,
            with_ground=False,
            range_crop_type="ego",
            use_gt_flow=False,
            log_subset=[sequence_id],
            flow_data_path=flow_folder,
        ),
    )
    assert (
        len(dataset) > 0
    ), f"No sequences found in {sequence_data_folder} with ID {sequence_id}."
    sequence = dataset[0]
    return sequence


# fmt: off

def fit_from_pc_list(
        exp_dir: Path,
    pc_list : list[np.ndarray],
    pc_masks : list[np.ndarray],
    options,   
):
    n_lidar_sweeps = len(pc_list)
      # ANCHOR: Initialize the trajectory field
    net = NeuralTrajField(traj_len=n_lidar_sweeps,
                filter_size=options.hidden_units,
                act_fn=options.act_fn, traj_type=options.traj_type, st_embed_type=options.st_embed_type)
    net.to(options.device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=options.lr, weight_decay=options.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200,400,600,800], gamma=0.5)
    
    # SECTION: Training steps
    min_loss = 1e10
    for i in tqdm.tqdm(range(options.iters)):
        # NOTE: Randomly sample  pc pairs
        do_val = np.mod(i, options.traj_len) == 0

        rnd_ids = np.random.choice(n_lidar_sweeps, options.traj_batch_size)
        
        # cur_metrics = {}
        for ref_id in rnd_ids:
            post_id = min(n_lidar_sweeps-1, ref_id+1)
            prev_id = max(0, ref_id-1)
            pc_ref = torch.from_numpy(pc_list[ref_id]).cuda()
            
            ref_traj_rt = net(pc_ref, ref_id, do_fwd_flow=True, do_bwd_flow=True, do_full_traj=True)

            pc_ref2post = net.transform_pts(ref_traj_rt['flow_fwd'], pc_ref)
            pc_ref2prev = net.transform_pts(ref_traj_rt['flow_bwd'], pc_ref)

            pc_prev = torch.from_numpy(pc_list[prev_id]).cuda()
            pc_post = torch.from_numpy(pc_list[post_id]).cuda()
            
            loss_chamfer_ref2prev, _ = my_chamfer_fn(pc_prev.unsqueeze(0), pc_ref2prev.unsqueeze(0), None, None)
            loss_chamfer_ref2post, _ = my_chamfer_fn(pc_post.unsqueeze(0), pc_ref2post.unsqueeze(0), None, None)

            post_traj_rt = net(pc_ref2post, post_id, do_fwd_flow=False, do_bwd_flow=True, do_full_traj=True)    
            prev_traj_rt = net(pc_ref2prev, prev_id, do_fwd_flow=True, do_bwd_flow=False, do_full_traj=True)
        
            loss_chamfer = loss_chamfer_ref2prev + loss_chamfer_ref2post
            
            # NOTE: Consistency loss
            # loss_traj_consist = ( (ref_traj_rt['traj'] - post_traj_rt['traj'])**2 ).mean() \
            #     + ( (ref_traj_rt['traj'] - prev_traj_rt['traj'])**2 ).mean()

            loss_consist = net.compute_traj_consist_loss(ref_traj_rt['traj'], post_traj_rt['traj'], pc_ref, pc_ref2post, ref_id, post_id, options.ctype) \
                + net.compute_traj_consist_loss(ref_traj_rt['traj'], prev_traj_rt['traj'], pc_ref, pc_ref2prev, ref_id, prev_id, options.ctype)

            tmp_id = n_lidar_sweeps // 2
            flow_ref2tmp = net.extract_flow(ref_id, tmp_id, ref_traj_rt['traj'])
            pc_ref2tmp = net.transform_pts(flow_ref2tmp, pc_ref)
            tmp_traj_rt = net(pc_ref2tmp, tmp_id, do_fwd_flow=False, do_bwd_flow=False, do_full_traj=True)
            loss_consist += net.compute_traj_consist_loss(ref_traj_rt['traj'], tmp_traj_rt['traj'], pc_ref, pc_ref2tmp, ref_id, tmp_id, options.ctype)
              
            loss = loss_chamfer + options.w_consist*loss_consist

            loss.backward()

         
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        logging.info(f"[Itr: {i}]"
                    f"[ref: {ref_id}]"
                    f"[loss: {loss:.5f}]"
                    f"[chamf_post: {loss_chamfer_ref2post:.4f}]"
                    f"[chamf_prev: {loss_chamfer_ref2prev:.4f}]"
                    f"[consist: {loss_consist:.4f}]")
        
    # Save flow results
    for ref_id in range(n_lidar_sweeps - 1):
        
        pc_np = pc_list[ref_id]
        pc_ref = torch.from_numpy(pc_np).cuda()
        ref_traj_rt = net(pc_ref, ref_id, do_fwd_flow=True, do_bwd_flow=True, do_full_traj=True)
        pc_ref2post = net.transform_pts(ref_traj_rt['flow_fwd'], pc_ref)
        flow_np = pc_ref2post.cpu().detach().numpy()

        flow_delta = flow_np - pc_np

        valid_mask = pc_masks[ref_id]

        full_flow = np.zeros((len(valid_mask), 3))
        full_flow[valid_mask] = flow_delta


        exp_dir / f"{ref_id:010d}.feather"

        df_dict = {
            "flow_tx_m": full_flow[:, 0],
            "flow_ty_m": full_flow[:, 1],
            "flow_tz_m": full_flow[:, 2],
            "is_valid": valid_mask,
        }
        df = pd.DataFrame(df_dict)
        save_feather(exp_dir / f"{ref_id:010d}.feather", df)

# fmt: on


@dataclass
class ConfigOptions:
    iters: int
    hidden_units: int
    act_fn: str
    traj_type: str
    st_embed_type: str
    w_consist: float
    ctype: str
    device: str
    lr: float
    weight_decay: float
    traj_batch_size: int
    traj_len: int


if __name__ == "__main__":

    # Take one argument: sequence_id
    args = argparse.ArgumentParser()
    args.add_argument("sequence_id", type=str)
    args = args.parse_args()

    sequence_id = args.sequence_id

    seq_res = load_sequence(
        sequence_data_folder=Path("/efs/argoverse2_seq_len_20/val"),
        flow_folder=Path("/efs/argoverse2_seq_len_20/val_sceneflow_feather"),
        sequence_length=20,
        sequence_id=sequence_id,
    )

    ego_pcs = [r.pc.global_pc.points.astype(np.float32) for r in seq_res]
    valid_masks = [r.pc.mask for r in seq_res]

    fit_from_pc_list(
        Path("/efs/argoverse2_seq_len_20/val_ntp_dagger/") / sequence_id,
        ego_pcs,
        valid_masks,
        options=ConfigOptions(
            iters=1000,
            hidden_units=128,
            act_fn="relu",
            traj_type="velocity",
            st_embed_type="cosine",
            w_consist=1,
            ctype="velocity",
            device="cuda:0",
            lr=0.003,
            weight_decay=0,
            traj_batch_size=4,
            traj_len=20,
        ),
    )
