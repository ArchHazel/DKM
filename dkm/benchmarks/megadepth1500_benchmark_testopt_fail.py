import numpy as np
import torch
from dkm.utils import *
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import cv2
import os
from math import atan2, degrees
import matplotlib.pyplot as plt
from pathlib import Path

matches_path = Path("assets/megadepth")
anno_path = Path("/mnt/c/Users/ivc-lab-5/Desktop/hazel/sam/results")

intrinsic_max_length = 1200

def rotation_matrix_to_ypr(R):
    pitch = atan2(-R[2,0], np.sqrt(R[0,0]**2 + R[1,0]**2))
    roll = atan2(R[1,0]/np.cos(pitch), R[0,0]/np.cos(pitch))
    yaw = atan2(R[2,1]/np.cos(pitch), R[2,2]/np.cos(pitch))

    pitch = degrees(pitch)
    roll = degrees(roll)
    yaw = degrees(yaw)
    return yaw, pitch, roll


class Megadepth1500BenchmarkTestSPEC:
    
    def __init__(self, data_root="data/megadepth", scene_names = None) -> None:
        if scene_names is None:
            self.scene_names = [
                "0015_0.1_0.3.npz",
                "0015_0.3_0.5.npz",
                "0022_0.1_0.3.npz",
                "0022_0.3_0.5.npz",
                "0022_0.5_0.7.npz",
            ]
            # in each npz file, there are 
            # 1. pair_infos 2. intrinsics 3.poses
        else:
            self.scene_names = scene_names
        self.scenes = [
            np.load(f"{data_root}/{scene}", allow_pickle=True)
            for scene in self.scene_names
        ]
        self.data_root = data_root

    def load_input(self, pairind, pairs, im_paths, intrinsics, poses, data_root):
        
        idx1, idx2 = pairs[pairind][0]
        K1 = intrinsics[idx1].copy()
        T1 = poses[idx1].copy()
        R1, t1 = T1[:3, :3], T1[:3, 3]
        K2 = intrinsics[idx2].copy()
        T2 = poses[idx2].copy()
        R2, t2 = T2[:3, :3], T2[:3, 3]
        R, t = compute_relative_pose(R1, t1, R2, t2)

        # read the images
        im1_path = f"{data_root}/{im_paths[idx1]}"
        im2_path = f"{data_root}/{im_paths[idx2]}"

        im1 = Image.open(im1_path)
        w1, h1 = im1.size
        im2 = Image.open(im2_path)
        w2, h2 = im2.size

        # scale the larger side to 1200 for each image
        scale1 = intrinsic_max_length / max(w1, h1)
        scale2 = intrinsic_max_length / max(w2, h2)
        w1, h1 = scale1 * w1, scale1 * h1
        w2, h2 = scale2 * w2, scale2 * h2

        # scale the intrinsics accordingly 
        K1[:2] = K1[:2] * scale1
        K2[:2] = K2[:2] * scale2



        return im1_path, im2_path, im1, im2, w1, h1, w2, h2, K1, K2, R, t, scale1, scale2

    def inference(self, model, im1_path, im2_path, w1, h1, w2, h2):
        
        dense_matches, dense_certainty, f_q_pyramid = model.match(im1_path, im2_path)
        
        # num of corresponding matches
        num_matches = 10000
        sparse_matches,_ = model.sample(
            dense_matches, dense_certainty, num_matches
        )

        kpts1 = sparse_matches[:, :2]
        kpts1 = (
            torch.stack(
                (
                    w1 * (kpts1[:, 0] + 1) / 2,
                    h1 * (kpts1[:, 1] + 1) / 2,
                ),
                axis=-1,
            )
        )
        kpts2 = sparse_matches[:, 2:]

        kpts2 = (
            torch.stack(
                (
                    w2 * (kpts2[:, 0] + 1) / 2,
                    h2 * (kpts2[:, 1] + 1) / 2,
                ),
                axis=-1,
            )
        )
        return kpts1, kpts2
    
    def Corr2Matrix(self,left_masks, right_masks, pA, pB):
        # Initialize the mask matrix with the size of (mask_num, mask_num)
        # mask_num is the number of masks
        # the row represents the mask of image A and the column represents the mask of image B 
        # initialize the mask matrix with empty lists
        MasksMatrix = [[[] for i in range(len(right_masks))] for j in range(len(left_masks))]

        # iterate through the keypoints correspondences
        for i in range(len(pA)):
            Ax , Ay = int(pA[i][0]) , int(pA[i][1])
            Bx , By = int(pB[i][0]), int(pB[i][1])

            # iterate through the masks to find the mask that contains the keypoints
            for idx, maskX in enumerate(left_masks):
                if maskX['segmentation'][Ay][Ax]:
                    for idy, maskY in enumerate(right_masks):
                        if maskY['segmentation'][By][Bx]:
                            # if the mask contains the keypoints, append the id of corrs 
                            MasksMatrix[idx][idy].append(i)

        return MasksMatrix


    
    def benchmark(self, model):

        res_dir = 'results/megadepth1500'
        os.makedirs(res_dir, exist_ok=True)

        # with no gradient when doing optimization

        with torch.no_grad():
            data_root = self.data_root
            tot_e_pose = []

            for scene_ind in range(len(self.scenes)):

                scene = self.scenes[scene_ind]
                pairs = scene["pair_infos"]
                intrinsics = scene["intrinsics"]
                poses = scene["poses"]
                im_paths = scene["image_paths"]
                pair_inds = range(len(pairs))


                # for pairind in tqdm(pair_inds):
                for pairind in pair_inds:
                    
                    idx1, idx2 = pairs[pairind][0]
                    im1_name = os.path.basename(im_paths[idx1])[:-4]
                    im2_name = os.path.basename(im_paths[idx2])[:-4]


                    # do inference on the chosen pair using name as the key
                    if im1_name != "3296298959_2bdd857a2e_o" or im2_name != "2362913561_1b797b1035_o":
                        continue
                    else:
                        # note here: the dimension w and h is not the size of the original images
                        im1_path, im2_path, im1, im2, w1, h1, w2, h2, K1, K2, R, t, scale1, scale2  = \
                            self.load_input(pairind, pairs, im_paths, intrinsics, poses, data_root)

                    # check if the match file exists, if no then do inference and save the matches
                    if not os.path.exists(matches_path/ f"{im1_name}_{im2_name}_matches.npy"):
                        # do inference on dkm
                        # kpts1 kpts2 they are resized to 1200 to calculate the errors 
                        kpts1, kpts2 = self.inference(model, im1_path, im2_path, w1, h1, w2, h2)
                        # kpts1_e kpts2_e they are the exact location of the keypoints in the original image
                        kpts1 = kpts1.cpu().numpy()
                        kpts2 = kpts2.cpu().numpy()
                        kpts1_e = kpts1 / scale1
                        kpts2_e = kpts2 / scale2
                        matches = np.hstack((kpts1_e,kpts2_e))
                        np.save(matches_path/ f"{im1_name}_{im2_name}_matches.npy", matches)
                    else:
                        matches = np.load(matches_path/ f"{im1_name}_{im2_name}_matches.npy",allow_pickle=True)
                        kpts1_e = matches[:,0:2]
                        kpts2_e = matches[:,2:4]
                        kpts1 = kpts1_e * scale1
                        kpts2 = kpts2_e * scale2

                    exp_dir = Path(f'exp_dir/megadepth/{im1_name}_{im2_name}')
                    if not exp_dir.exists():
                        os.mkdir(exp_dir)

                    # random give 50 index to kpt1 and kpt2
                    ran_i = np.random.randint(kpts1_e.shape[0],size=50)


                    kpt1 = kpts1[ran_i]
                    kpt2 = kpts2[ran_i]






                    # debug
                    kpt1_c = Kinv(kpt1, K1)
                    kpt1_homo = np.concatenate([kpt1_c, np.ones((kpt1_c.shape[0], 1))], axis=1)
                    
                    # kpt1_homo2 = np.concatenate([kpt1, np.ones((kpt1.shape[0], 1))], axis=1)
                    # kpt1_c2 = (np.linalg.inv(K1) @ kpt1_homo2.T).T
                    # kpt1_c2 /= kpt1_c2[:, 2]

                    kpt2_rt = np.einsum('ij, kj -> ki', R, kpt1_homo) + t
                    print(t)


                    kpt2_p = np.einsum('ij, kj -> ki', K2, kpt2_rt)

                    kpt_diff = kpt2 - kpt2_p[:, :2] / kpt2_p[:, 2:3]
                    print(kpt_diff)

                    kpts1_vis = [cv2.KeyPoint(x[0], x[1], 1) for x in kpt1 / scale1] 
                    kpts2_vis = [cv2.KeyPoint(x[0], x[1], 1) for x in kpt2_p[:, :2] / kpt2_p[:, 2:3] / scale2] 
                    kpts2_kp_vis = [cv2.KeyPoint(x[0], x[1], 1) for x in kpt2 / scale2]
                    matches = [cv2.DMatch(i, i, 1) for i in range(len(kpt1))]

                    img1 = cv2.cvtColor(np.array(im1), cv2.COLOR_RGB2BGR)
                    img2 = cv2.cvtColor(np.array(im2), cv2.COLOR_RGB2BGR)

                    matched_img = cv2.drawMatches(img2, kpts2_kp_vis, img2, kpts2_vis, matches, None)
                    cv2.imwrite(str(exp_dir / f'corrB_corrARt.png'), matched_img)

                    matched_img = cv2.drawMatches(img1, kpts1_vis, img2, kpts2_kp_vis, matches, None)
                    cv2.imwrite(str(exp_dir / f'corrA_corrB.png'), matched_img)

                    matched_img = cv2.drawMatches(img1, kpts1_vis, img2, kpts2_vis, matches, None)
                    cv2.imwrite(str(exp_dir / f'corrA_corrARt.png'), matched_img)



                
                    
                    


                    left_masks = np.load(anno_path/f'{im1_name}.npy', allow_pickle = True)
                    right_masks = np.load(anno_path/f'{im2_name}.npy', allow_pickle = True)

                    

                    if not exp_dir.exists():
                        exp_dir.mkdir(parents=True, exist_ok=True)
                        
                        # left masks
                        trans_mask_total = np.ones((left_masks[0]['segmentation'].shape[0], left_masks[0]['segmentation'].shape[1], 4))
                        left_masks_sorted = sorted(left_masks, key=(lambda x: x['area']), reverse=True)
                        for mask_id, mask in enumerate(left_masks_sorted):
                            trans_mask = np.ones((mask['segmentation'].shape[0], mask['segmentation'].shape[1], 4))
                            trans_mask[:,:,3] = 0
                            m = mask['segmentation']
                            color_mask = np.concatenate([np.random.random(3), [0.35]])
                            trans_mask[m] = color_mask
                            cv2.imwrite(str(exp_dir / f'left_{mask_id}.png'), (trans_mask * 255).astype(np.uint8))

                            trans_mask_total[m] = color_mask
                        cv2.imwrite(str(exp_dir / f'left_total.png'), (trans_mask_total * 255).astype(np.uint8))

                        # right masks
                        trans_mask_total = np.ones((right_masks[0]['segmentation'].shape[0], right_masks[0]['segmentation'].shape[1], 4))
                        right_masks_sorted = sorted(right_masks, key=(lambda x: x['area']), reverse=True)
                        for mask_id, mask in enumerate(right_masks_sorted):
                            trans_mask = np.ones((mask['segmentation'].shape[0], mask['segmentation'].shape[1], 4))
                            trans_mask[:,:,3] = 0
                            m = mask['segmentation']
                            color_mask = np.concatenate([np.random.random(3), [0.35]])
                            trans_mask[m] = color_mask
                            cv2.imwrite(str(exp_dir / f'right_{mask_id}.png'), (trans_mask * 255).astype(np.uint8))
                            trans_mask_total[m] = color_mask
                        cv2.imwrite(str(exp_dir / f'right_total.png'), (trans_mask_total * 255).astype(np.uint8))


                    MaskMatrix = self.Corr2Matrix(left_masks, right_masks, kpts1_e, kpts2_e)


                    paired_corr = {}
                    # the key is "correspondences within similar semantic backgrounds" 
                    # the value is the occurence times 
                    for ia in range(len(MaskMatrix)):
                        for ib in range(len(MaskMatrix[0])):
                            if(len(MaskMatrix[ia][ib]) >= 20):
                                correspondence = tuple(MaskMatrix[ia][ib])
                                if paired_corr.get(correspondence) is None:
                                    paired_corr[correspondence] = [(ia, ib)]
                                else:
                                    paired_corr[correspondence].append((ia, ib))



                    # show the gt yaw pitch roll scatter point
                    fig = plt.figure()
                    ax = fig.add_subplot(projection='3d')
                    yaw, pitch, roll = rotation_matrix_to_ypr(R)
                    ax.scatter(yaw, pitch, roll, marker='^')
                    plt.show()



                    R_para = []
                    t_para = []



                    for corr_idx, corr in enumerate(paired_corr):
                        print('mask pair id', corr)

                        # create random idx with the same length as corr
                        ran_i = np.random.randint(kpts1_e.shape[0],size=len(corr))
                        kpt1_r = kpts1[ran_i] 
                        kpt2_r = kpts2[ran_i] 

                        kpt1 = kpts1[np.array(corr, dtype=np.int32)] 
                        kpt2 = kpts2[np.array(corr, dtype=np.int32)] 

                        kpt1 = kpt1_r
                        kpt2 = kpt2_r




                        norm_threshold = 0.5 / (
                                    np.mean(np.abs(K1[:2, :2])) + np.mean(np.abs(K2[:2, :2])))
                        
                        R_est, t_est, mask = estimate_pose(
                                    kpt1,
                                    kpt2,
                                    K1,
                                    K2,
                                    norm_threshold,
                                    conf=0.99999,
                                    method=cv2.LMEDS
                                    )








                        
                        tot_e_t_local, tot_e_R_local, tot_e_pose_local = [], [], []

                        print("start est for corr:", len(corr),"repeat for five times")

                        # same corr, repeat
                        replicate = 5
                        for _ in range(replicate):
                            shuffleKeypoints = True
                            if shuffleKeypoints:
                                shuffling = np.random.permutation(np.arange(len(kpt1)))
                                kpt1 = kpt1[shuffling]
                                kpt2 = kpt2[shuffling]

                                kpt1_r = kpt1_r[shuffling]
                                kpt2_r = kpt2_r[shuffling]
                            try:
                                norm_threshold = 0.5 / (
                                    np.mean(np.abs(K1[:2, :2])) + np.mean(np.abs(K2[:2, :2])))
                                R_est, t_est, mask = estimate_pose(
                                    kpt1,
                                    kpt2,
                                    K1,
                                    K2,
                                    norm_threshold,
                                    conf=0.99999,
                                    method=cv2.LMEDS
                                    )
                                
                                R_est_r, t_est_r, mask_r = estimate_pose(
                                    kpt1_r,
                                    kpt2_r,
                                    K1,
                                    K2,
                                    norm_threshold,
                                    conf=0.99999,
                                    )




                                print("R t comparisons\n", 'R', R_est, R_est_r, '\nt', t_est, t_est_r)
                                # print('reconstruct kpt diff', kpt_diff.mean(axis=0), kpt_r_diff.mean(axis=0))
                                T1_to_2_est = np.concatenate((R_est, t_est), axis=-1)  
                                T1_to_2_est_r = np.concatenate((R_est_r, t_est_r), axis=-1)

                                e_t, e_R = compute_pose_error(T1_to_2_est, R, t)
                                e_t_r, e_R_r = compute_pose_error(T1_to_2_est_r, R, t)
                                
                                yaw_est, pitch_est, roll_est = rotation_matrix_to_ypr(R_est)
                                yaw_est_r, pitch_est_r, roll_est_r = rotation_matrix_to_ypr(R_est_r)

                                # print("gt Yaw:", yaw, "est Yaw:", yaw_est)
                                # print("gt Pitch:", pitch, "est Pitch:", pitch_est)
                                # print("gt Roll:", roll, "est Roll:", roll_est)
                                est = [yaw_est, pitch_est, roll_est]

                                ax.scatter(yaw_est, pitch_est, roll_est, marker='o')
                                ax.scatter(yaw_est_r, pitch_est_r, roll_est_r, marker='*')
                                R_para.append(est)
                                
                                e_pose = max(e_t, e_R)
                                # print(e_pose)
                            except Exception as e:
                                print(repr(e),"Should happen some exception")
                                e_t, e_R = 90, 90
                                e_pose = max(e_t, e_R)

                            tot_e_t_local.append(e_t)
                            tot_e_R_local.append(e_R)     
                            tot_e_pose_local.append(e_pose) 
                            
                            
                        
                        
                        img1 = cv2.cvtColor(np.array(im1), cv2.COLOR_RGB2BGR)
                        img2 = cv2.cvtColor(np.array(im2), cv2.COLOR_RGB2BGR)



                        kpts1_vis = [cv2.KeyPoint(x[0], x[1], 1) for x in kpt1 / scale1] 
                        kpts2_vis = [cv2.KeyPoint(x[0], x[1], 1) for x in kpt2 / scale2] 
                        matches = [cv2.DMatch(i, i, 1) for i in range(len(kpt1))]
                        matched_img = cv2.drawMatches(img1, kpts1_vis, img2, kpts2_vis, matches, None)
                        cv2.imwrite(str(exp_dir / f'matched_img_{corr_idx}.png'), matched_img)


                        print("finish one set of corrs, results are:")
                        print('tot_e_t_local', tot_e_t_local)
                        print('tot_e_R_local', tot_e_R_local)
                        print('tot_e_pose_local', tot_e_pose_local)
                        print('err', np.array(tot_e_t_local).mean(), np.array(tot_e_R_local).mean(), np.array(tot_e_pose_local).mean())
    
            
            tot_e_pose = np.array(tot_e_pose)
            thresholds = [5, 10, 20]
            auc = pose_auc(tot_e_pose, thresholds)
            acc_5 = (tot_e_pose < 5).mean()
            acc_10 = (tot_e_pose < 10).mean()
            acc_15 = (tot_e_pose < 15).mean()
            acc_20 = (tot_e_pose < 20).mean()
            map_5 = acc_5
            map_10 = np.mean([acc_5, acc_10])
            map_20 = np.mean([acc_5, acc_10, acc_15, acc_20])
            return {
                "auc_5": auc[0],
                "auc_10": auc[1],
                "auc_20": auc[2],
                "map_5": map_5,
                "map_10": map_10,
                "map_20": map_20,
            }
