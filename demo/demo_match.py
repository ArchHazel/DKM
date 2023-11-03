from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from dkm.utils.utils import tensor_to_pil
import cv2 
import os

from dkm import DKMv3_outdoor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":

    def convert_coordinates( query_coords, query_to_support, wq, hq, wsup, hsup):
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

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--im_A_path", default="assets/2464627575_604e98f303_o.jpg", type=str)
    parser.add_argument("--im_B_path", default="assets/2464628195_1a5936f331_o.jpg", type=str)
    parser.add_argument("--save_path", default="demo/dkmv3_warp_sacre_coeur.jpg", type=str)

    args, _ = parser.parse_known_args()
    im1_path = args.im_A_path
    im2_path = args.im_B_path
    save_path = args.save_path

    # Create model
    dkm_model = DKMv3_outdoor(device=device)

    H, W = 864, 1152

    im1 = Image.open(im1_path).resize((W, H))
    im2 = Image.open(im2_path).resize((W, H))

    # Match
    warp, certainty, f = dkm_model.match(im1_path, im2_path, device=device)

    # Sampling not needed, but can be done with model.sample(warp, certainty)
    # choose the best #(num) among the corresponding matches
    good_matches, _ = dkm_model.sample(warp, certainty,num=100)

    im1 = Image.open(im1_path)
    im2 = Image.open(im2_path)
    
    w1, h1 = Image.open(im1_path).size
    w2, h2 = Image.open(im2_path).size
    print(good_matches[:,:2][:,0].shape)

    pos_a, pos_b = convert_coordinates(
                    good_matches[:, :2], good_matches[:, 2:], w1, h1, w2, h2
                )
    
    img1 = cv2.cvtColor(np.array(im1), cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(np.array(im2), cv2.COLOR_RGB2BGR)
    kpts1 = [cv2.KeyPoint(x[0], x[1], 1) for x in pos_a]
    kpts2 = [cv2.KeyPoint(x[0], x[1], 1) for x in pos_b]
    np.save("2464627575_604e98f303_o_2464628195_1a5936f331_o_matches.npy",[pos_a,pos_b])

    matches = [cv2.DMatch(i, i, 1) for i in range(len(pos_a))]
    matched_img = cv2.drawMatches(img1, kpts1, img2, kpts2, matches, None)


    res_dir = "assets"
    test_name = "keypointOnPicture"
    print('match path', os.path.join(res_dir, f'{test_name}.png'))
    cv2.imwrite(os.path.join(res_dir, f'{test_name}.png'), matched_img)

    x1 = (torch.tensor(np.array(im1)) / 255).to(device).permute(2, 0, 1)
    x2 = (torch.tensor(np.array(im2)) / 255).to(device).permute(2, 0, 1)

    im2_transfer_rgb = F.grid_sample(
    x2[None], warp[:,:W, 2:][None], mode="bilinear", align_corners=False
    )[0]
    im1_transfer_rgb = F.grid_sample(
    x1[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
    )[0]
    warp_im = torch.cat((im2_transfer_rgb,im1_transfer_rgb),dim=2)
    white_im = torch.ones((H,2*W),device=device)
    vis_im = certainty * warp_im + (1 - certainty) * white_im
    tensor_to_pil(vis_im, unnormalize=False).save(save_path)
