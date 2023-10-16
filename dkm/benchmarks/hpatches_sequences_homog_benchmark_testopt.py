from PIL import Image
import numpy as np

import os
import torch
from torch.nn import functional as F
from tqdm import tqdm
from dkm.utils import pose_auc
import cv2
from testopt.model import TestOpt


class HpatchesHomogBenchmarkTestOpt:
    """Hpatches grid goes from [0,n-1] instead of [0.5,n-0.5]"""

    def __init__(self, dataset_path) -> None:
        seqs_dir = "hpatches-sequences-release"
        self.seqs_path = os.path.join(dataset_path, seqs_dir)
        self.seq_names = sorted(os.listdir(self.seqs_path))
        # Ignore seqs is same as LoFTR.
        self.ignore_seqs = set(
            [
                "i_contruction",
                "i_crownnight",
                "i_dc",
                "i_pencils",
                "i_whitebuilding",
                "v_artisans",
                "v_astronautis",
                "v_talent",
            ]
        )

    def convert_coordinates(self, query_coords, query_to_support, wq, hq, wsup, hsup):
        offset = 0.5  # Hpatches assumes that the center of the top-left pixel is at [0,0] (I think)
        query_coords = query_coords.cpu()
        query_to_support = query_to_support.cpu()
        query_coords = (
            np.stack(
                (
                    wq * (query_coords[..., 0] + 1) / 2,
                    hq * (query_coords[..., 1] + 1) / 2,
                ),
                axis=-1,
            )
            - offset
        )
        query_to_support = (
            np.stack(
                (
                    wsup * (query_to_support[..., 0] + 1) / 2,
                    hsup * (query_to_support[..., 1] + 1) / 2,
                ),
                axis=-1,
            )
            - offset
        )
        return query_coords, query_to_support

    def benchmark(self, model, model_name = None):
        n_matches = []
        homog_dists = []

        res_dir = 'results/hpatches_finetune'
        os.makedirs(res_dir, exist_ok=True)

        for seq_idx, seq_name in tqdm(
            enumerate(self.seq_names), total=len(self.seq_names)
        ):
            if seq_name in self.ignore_seqs:
                continue
            im1_path = os.path.join(self.seqs_path, seq_name, "1.ppm")
            im1 = Image.open(im1_path)
            w1, h1 = im1.size
            for im_idx in range(2, 7):
                im2_path = os.path.join(self.seqs_path, seq_name, f"{im_idx}.ppm")
                im2 = Image.open(im2_path)

                test_name = f'{seq_name}_1_{im_idx}'

                w2, h2 = im2.size
                H = np.loadtxt(
                    os.path.join(self.seqs_path, seq_name, "H_1_" + str(im_idx))
                )
                dense_matches, dense_certainty, f_q_pyramid = model.match(
                    im1_path, im2_path
                )
                match_num = 100 # 5000
                good_matches, _ = model.sample(dense_matches, dense_certainty, match_num)
                # pos_a, pos_b = self.convert_coordinates(
                #     good_matches[:, :2], good_matches[:, 2:], w1, h1, w2, h2
                # )

                feat4 = F.interpolate(f_q_pyramid[4], f_q_pyramid[2].shape[-2:], mode="bilinear")
                feats = torch.cat([feat4, f_q_pyramid[2]], dim=1)
                testopt = TestOpt(good_matches, feats[0], feats[1], (h1, w1))
                testopt.fit()
                pos_a, pos_b = testopt.eval(h2, w2)
                
                
                # viz
                # img1 = cv2.cvtColor(np.array(im1), cv2.COLOR_RGB2BGR)
                # img2 = cv2.cvtColor(np.array(im2), cv2.COLOR_RGB2BGR)
                # kpts1 = [cv2.KeyPoint(x[0], x[1], 1) for x in pos_a]
                # kpts2 = [cv2.KeyPoint(x[0], x[1], 1) for x in pos_b]

                # matches = [cv2.DMatch(i, i, 1) for i in range(len(pos_a))]
                # matched_img = cv2.drawMatches(img1, kpts1, img2, kpts2, matches, None)
                # print('match path', os.path.join(res_dir, f'{test_name}.png'))
                # cv2.imwrite(os.path.join(res_dir, f'{test_name}.png'), matched_img)


                # sift = cv2.SIFT_create()
                # find the keypoints and descriptors with SIFT
                # kp1, des1 = sift.detectAndCompute(img1,None)
                # kp2, des2 = sift.detectAndCompute(img2,None)
                # bf = cv2.BFMatcher()
                # matches = bf.knnMatch(des1,des2,k=2)
                
                # pos_a = pos_a.astype(np.int32).tolist()
                # pos_b = pos_b.astype(np.int32).tolist()
                # print(pos_a)
                
                try:
                    H_pred, inliers = cv2.findHomography(
                        pos_a,
                        pos_b,
                        method = cv2.RANSAC,
                        confidence = 0.99999,
                        ransacReprojThreshold = 3 * min(w2, h2) / 480,
                    )
                except:
                    H_pred = None
                if H_pred is None:
                    H_pred = np.zeros((3, 3))
                    H_pred[2, 2] = 1.0

                    print(H_pred, 'H_pred is None by homography')

                # print('Postprocessing')

                corners = np.array(
                    [[0, 0, 1], [0, h1 - 1, 1], [w1 - 1, 0, 1], [w1 - 1, h1 - 1, 1]]
                )
                real_warped_corners = np.dot(corners, np.transpose(H))
                real_warped_corners = (
                    real_warped_corners[:, :2] / real_warped_corners[:, 2:]
                )
                warped_corners = np.dot(corners, np.transpose(H_pred))
                warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
                mean_dist = np.mean(
                    np.linalg.norm(real_warped_corners - warped_corners, axis=1)
                ) / (min(w2, h2) / 480.0)
                homog_dists.append(mean_dist)
                print('mean_dist', mean_dist)
        n_matches = np.array(n_matches)
        thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        auc = pose_auc(np.array(homog_dists), thresholds)
        return {
            "hpatches_homog_auc_3": auc[2],
            "hpatches_homog_auc_5": auc[4],
            "hpatches_homog_auc_10": auc[9],
        }
