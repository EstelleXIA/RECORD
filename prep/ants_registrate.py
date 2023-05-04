import ants
import json
import pandas as pd
import os
import shutil
from datetime import datetime
import argparse
import SimpleITK as sitk
import torch
from monai.networks.blocks import Warp
import nibabel as nib
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--input_img', help='Input folder for serial CT images', required=True)
parser.add_argument('--input_mask', help='Input folder for serial segmentation model predicted masks', required=True)
parser.add_argument('--json_file', help='Path of .json file with baseline-follow-up pairs',
                    default="ants_json.json", required=True)
parser.add_argument('--output', help='Output folder for forward and inverse transforms', required=True)
parser.add_argument('--maskpath', help='Registered masks for difference map generation',
                    default="data/mask_ants/", required=True)
parser.add_argument('--check', action="store_true",
                    help='Registration depends on initialization, may have failed.'
                         'If DICE lower than 0.9, suggest to re-run this sample.')


args = parser.parse_args()
fwdtransforms_p = os.path.join(args.output, "fwdtransforms")
invtransforms_p = os.path.join(args.output, "invtransforms")
composite_p = os.path.join(args.output, "invtransforms_composite")
transformed_img_p = os.path.join(args.output, "image_ants")
transformed_mask_p = os.path.join(args.output, "mask_ants")
copy_mask_path = args.maskpath

if args.json_file:
    with open(args.json_file, "r") as ff:
        pair_json = json.load(ff)
        pair_json = pd.DataFrame(pair_json)
else:
    input_img_path = args.input_img
    input_mask_path = args.input_mask
    assert len(os.listdir(input_img_path)) == len(os.listdir(input_mask_path))
    files = sorted(os.listdir(input_img_path))
    patients = list(set(list(map(lambda x: x[:10], files))))
    pair_json = []
    for patient in patients:
        patient_visits = list(filter(lambda x: x.startswith(patient), files))
        baseline = min(patient_visits, key=lambda visit: datetime.strptime(visit[11:19], "%Y%m%d"))
        patient_visits.remove(baseline)
        for followup in patient_visits:
            pair_json.append(
                {"image": os.path.join(input_img_path, baseline),
                 "label": os.path.join(input_mask_path, baseline.replace("_0000", "")),
                 "image_1": os.path.join(input_img_path, followup),
                 "label_1": os.path.join(input_mask_path, followup.replace("_0000", ""))}
                )
        with open("ants_json.json", "w") as f:
            json.dump(pair_json, f)
        pair_json = pd.DataFrame(pair_json)

for idx in range(pair_json.nrows):
    f = pair_json.loc[idx, "image"]
    fm = pair_json.loc[idx, "label"]
    m = pair_json.loc[idx, "image_1"]
    mm = pair_json.loc[idx, "label_1"]

    invtransform_mat = os.path.join(invtransforms_p, f"{os.path.basename(f)[:19]}_{os.path.basename(m)[:19]}.mat")
    invtransform_nii_gz = os.path.join(invtransforms_p, f"{os.path.basename(f)[:19]}_{os.path.basename(m)[:19]}.nii.gz")

    fwdtransform_mat = os.path.join(fwdtransforms_p, f"{os.path.basename(f)[:19]}_{os.path.basename(m)[:19]}.mat")
    fwdtransform_nii_gz = os.path.join(fwdtransforms_p, f"{os.path.basename(f)[:19]}_{os.path.basename(m)[:19]}.nii.gz")

    transformed_image = os.path.join(transformed_img_p, os.path.basename(m))
    transformed_mask = os.path.join(transformed_img_p, os.path.basename(mm))

    shutil.copy(f, transformed_img_p)
    shutil.copy(fm, transformed_mask_p)

    fix_img = ants.image_read(f)
    move_img = ants.image_read(m)
    move_img.set_origin(fix_img.origin)
    move_mask = ants.image_read(mm)
    move_mask.set_origin(fix_img.origin)

    mytx = ants.registration(fixed=fix_img,  moving=move_img, type_of_transform='SyN', verbose=False)
    m_wrap200 = ants.apply_transforms(fixed=fix_img, moving=move_img,
                                      transformlist=mytx['fwdtransforms'], defaultvalue=-200)
    m_wrap200.to_file(transformed_image)

    mm_wrap200 = ants.apply_transforms(fixed=fix_img, moving=move_img,
                                       transformlist=mytx['fwdtransforms'], defaultvalue=0)
    mm_wrap200.to_file(transformed_mask)

    assert (len(mytx["invtransforms"]) == 2) and (mytx["invtransforms"][0].endswith(".mat")) and (mytx["invtransforms"][1].endswith(".nii.gz"))
    assert (len(mytx["fwdtransforms"]) == 2) and (mytx["fwdtransforms"][0].endswith(".nii.gz")) and (mytx["fwdtransforms"][1].endswith(".mat"))

    shutil.copy(mytx["invtransforms"][0], os.path.join(invtransforms_p, invtransform_mat))
    shutil.copy(mytx["invtransforms"][1], os.path.join(invtransforms_p, invtransform_nii_gz))

    shutil.copy(mytx["fwdtransforms"][0], os.path.join(fwdtransforms_p, fwdtransform_nii_gz))
    shutil.copy(mytx["fwdtransforms"][1], os.path.join(fwdtransforms_p, fwdtransform_mat))

    shutil.copy(transformed_mask, copy_mask_path)

    # composite inverse transforms
    elastix_def_img = sitk.ReadImage(invtransform_nii_gz, sitk.sitkVectorFloat64)
    displacement_field_transform = sitk.DisplacementFieldTransform(elastix_def_img)
    affine_transform = sitk.AffineTransform(sitk.ReadTransform(invtransform_mat)).GetInverse()

    composite_transform = sitk.CompositeTransform([displacement_field_transform,
                                                   affine_transform])

    sitk_mask = sitk.ReadImage(mm)
    sitk_moving_file = sitk.ReadImage(m)
    sitk_ref = sitk.ReadImage(f)
    sitk_mask.SetOrigin(sitk_ref.GetOrigin())
    sitk_moving_file.SetOrigin(sitk_ref.GetOrigin())

    sitk_moving_file_reg = sitk.ReadImage(transformed_mask)

    sitk_inv_reg_mask = sitk.Resample(sitk_moving_file_reg, sitk_moving_file,
                                      composite_transform,
                                      sitk.sitkNearestNeighbor, 0.0, 3)

    displ = sitk.TransformToDisplacementField(composite_transform,
                                              sitk.sitkVectorFloat64,
                                              sitk_moving_file.GetSize(),
                                              sitk_moving_file.GetOrigin(),
                                              sitk_moving_file.GetSpacing(),
                                              sitk_moving_file.GetDirection())

    sitk.WriteImage(displ, os.path.join(composite_p, os.path.basename(invtransform_nii_gz)))

    if args.check:
        warp_nearest = Warp(mode="nearest", padding_mode="zeros")
        warp_linear = Warp(mode="bilinear", padding_mode="border")

        torch_mask = torch.from_numpy(nib.load(transformed_mask).get_fdata()).unsqueeze(0).unsqueeze(0)
        torch_ddf = torch.from_numpy(sitk.GetArrayFromImage(displ).
                                     transpose((2, 1, 0, 3))).unsqueeze(0).swapaxes(0, -1)[:, :, :, :, 0].unsqueeze(0)
        torch_ddf_div = torch_ddf / torch.tensor([2.0, 2.0, 2.0]).view(1, -1, 1, 1, 1)
        torch_reg_mask_inv_div = warp_nearest(torch_mask, torch_ddf_div)[0, 0]

        torch_raw_dice = np.bitwise_and(torch_reg_mask_inv_div.numpy().astype(bool),
                                        sitk.GetArrayFromImage(sitk_mask).transpose((2, 1, 0)).astype(bool)).sum() / \
                         np.bitwise_or(torch_reg_mask_inv_div.numpy().astype(bool),
                                       sitk.GetArrayFromImage(sitk_mask).transpose((2, 1, 0)).astype(bool)).sum()
        torch_sitk_dice = np.bitwise_and(torch_reg_mask_inv_div.numpy().astype(bool),
                                         sitk.GetArrayFromImage(sitk_inv_reg_mask).transpose((2, 1, 0)).astype(bool)).sum() / \
                          np.bitwise_or(torch_reg_mask_inv_div.numpy().astype(bool),
                                        sitk.GetArrayFromImage(sitk_inv_reg_mask).transpose((2, 1, 0)).astype(bool)).sum()

        sum_ratio = torch_reg_mask_inv_div.numpy().astype(bool).sum() / sitk.GetArrayFromImage(sitk_mask).transpose(
            (2, 1, 0)).astype(bool).sum()
        sum_total = sitk.GetArrayFromImage(sitk_mask).transpose((2, 1, 0)).astype(bool).sum()
        if sum_total > 1000 and torch_raw_dice < 0.9 and abs(sum_ratio-1) > 0.1:
            print(f"Suggest to re-run {os.path.basename(pair_json.loc[idx, 'image'])},"
                  f" DICE {torch_raw_dice}, ratio {sum_ratio}!")
