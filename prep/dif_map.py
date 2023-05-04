import nibabel as nib
import numpy as np
import os
from datetime import datetime
import cc3d
from tqdm import tqdm
from copy import deepcopy
import SimpleITK as sitk
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Input folder for serial predicted masks', required=True)
parser.add_argument('--output', help='Output folder for empirically adjusted difference maps', required=True)
parser.add_argument('--livermask', help='Add anatomical location information')

args = parser.parse_args()

base_pred = args.input
save_refine = args.output
livermask_path = args.livermask

files = sorted(os.listdir(base_pred))
patients = list(set(list(map(lambda x: x[:10], files))))
if not os.path.exists(save_refine):
    os.makedirs(save_refine)


def cal_dice(previous, current):
    dice = np.bitwise_and(abs(previous).astype(bool), abs(current).astype(bool)).sum() / \
           np.bitwise_or(abs(previous).astype(bool), abs(current).astype(bool)).sum()
    return dice


def get_suspected_fp(dif_map_cc3d, v_suspected=25000):
    """
    This is used for detecting possible false positives in the difference map
    :param dif_map_cc3d: followup dif_map, X * Y * Z
    :param v_suspected: suspected volume of false positive, another 150000 is used to avoid the impact of fusion
    :return: voxel counts & bbox of suspected FPs, voxel counts: a list of numerical values,
    bbox: a list of [x_min, x_max, y_min, y_max, z_min, z_max], candidate_idx: lesion number.
    """
    cc3d_stats = cc3d.statistics(dif_map_cc3d)
    candidates_idx = [x + 1 for x in range(cc3d_stats["voxel_counts"][1:].shape[0])
                      if v_suspected < cc3d_stats["voxel_counts"][1:][x] < 150000]
    voxel_counts_candidates, bbox_candidates = None, None
    if candidates_idx:
        voxel_counts_candidates = [cc3d_stats["voxel_counts"][x] for x in candidates_idx]
        bboxes = [cc3d_stats["bounding_boxes"][x] for x in candidates_idx]
        bbox_candidates = [[bboxes[i][0].start, bboxes[i][0].stop,
                            bboxes[i][1].start, bboxes[i][1].stop,
                            bboxes[i][2].start, bboxes[i][2].stop, ] for i in range(len(bboxes))]
    return voxel_counts_candidates, bbox_candidates, candidates_idx


def check_true_rapid_change(bboxes, baseline_mask, followup_mask, threshold_dice=0.1):
    """
    Used to check whether is a drastic change lesion (FPs) or not
    :param bboxes: from 'get_suspected_fp'
    :param baseline_mask: input mask of baseline
    :param followup_mask: input mask of followup
    :param threshold_dice: the overlap of roi / bbox
    :return: checked box (remove not FPs)
    """
    if bboxes:
        check_boxes = deepcopy(bboxes)
        for box in bboxes:
            baseline_roi = baseline_mask[int(box[0]):int(box[1]), int(box[2]):int(box[3]), int(box[4]):int(box[5])]
            followup_roi = followup_mask[int(box[0]):int(box[1]), int(box[2]):int(box[3]), int(box[4]):int(box[5])]
            if cal_dice(baseline_roi, followup_roi) > threshold_dice:
                check_boxes.remove(box)
        return None if len(check_boxes) == 0 else check_boxes


def check_true_fps(dif_map_current, bboxes, ids, dif_map_restore, threshold_var=1000):
    """
    Check whether a suspected FP in get_suspected_fp needs to remove
    :param dif_map_current: Current dif_map after cc3d
    :param bboxes: suspected bboxes from get_suspected_fp
    :param ids: lesion ids in current dif_map
    :param dif_map_restore: dif_maps from time one to current (1~n)
    :param threshold_var: threshold of second derivative variance
    :return: refined current dif_map.
    """
    if bboxes:
        for i, box in enumerate(bboxes):
            volumes_dif = np.array(dif_map_restore)[:, int(box[0]):int(box[1]),
                                                    int(box[2]):int(box[3]),
                                                    int(box[4]):int(box[5])].sum((1, 2, 3))
            # at least three time pairs, if only two, duplicate the followup1 (baseline + followup1 + followup1)
            volumes_dif_visits = (np.repeat(volumes_dif, 2) if volumes_dif.shape[0] == 1 else volumes_dif)
            voxel_absolute_var = (np.diff(np.append([0], volumes_dif_visits))).var()
            if voxel_absolute_var > threshold_var:
                print(f"Trigger variance for {box}, Variance is {voxel_absolute_var}")
                dif_map_current[dif_map_current == ids[i]] = 0
        dif_map_current[dif_map_current > 0] = 1
    return dif_map_current


def remove_by_livermask(refined_dif_map, baseline_visit_name, followup_visit_name,
                        liver_mask_path, tolerance_liver=0.0):
    """
    To remove different areas outside the liver for both pos_map and neg_map
    :param refined_dif_map: Difference map after empirical FPs remove
    :param baseline_visit_name: baseline CT scan
    :param followup_visit_name: followup CT scan
    :param liver_mask_path: livermask root
    :param tolerance_liver: overlap between dif_lesion and livermask, if a lesion is inside the liver, always=1
    :return: removed difference outside the liver
    """

    baseline_liver = nib.load(os.path.join(liver_mask_path, baseline_visit_name)).get_fdata().astype(bool)
    followup_liver = nib.load(os.path.join(liver_mask_path, followup_visit_name)).get_fdata().astype(bool)
    livermask = np.bitwise_or(baseline_liver, followup_liver).astype(bool)
    refined_dif_map_cc3d, foreground_num = cc3d.connected_components(refined_dif_map, return_N=True)
    for idx in range(1, foreground_num + 1):
        extracted_image = refined_dif_map_cc3d * (refined_dif_map_cc3d == idx)
        extracted_image[extracted_image > 0] = 1
        if np.bitwise_and(extracted_image.astype(bool), livermask).sum()/extracted_image.sum() <= tolerance_liver:
            print(f"{followup_visit_name}, Trigger livermask!!!")
            refined_dif_map_cc3d[refined_dif_map_cc3d == idx] = 0
    refined_dif_map_cc3d[refined_dif_map_cc3d > 0] = 1
    return refined_dif_map_cc3d


def refine_mask(raw_dif_map, dif_map_restore, baseline_mask, followup_mask, baseline_name, followup_name,
                liver_mask_path, connect=6, v_suspected=10000, threshold_var=int(100000000*0.4*0.4),
                threshold_dice=0.04, tolerance_liver=0.001):
    dif_cc3d = cc3d.connected_components(raw_dif_map, connectivity=connect)
    voxel, bbox, lesion_ids = get_suspected_fp(dif_map_cc3d=dif_cc3d, v_suspected=v_suspected)
    bbox = check_true_rapid_change(bboxes=bbox, baseline_mask=baseline_mask, followup_mask=followup_mask,
                                   threshold_dice=threshold_dice)
    dif_current = check_true_fps(dif_map_current=dif_cc3d, bboxes=bbox, ids=lesion_ids,
                                 dif_map_restore=dif_map_restore, threshold_var=threshold_var)
    if livermask_path:
        dif_current = remove_by_livermask(refined_dif_map=dif_current, baseline_visit_name=baseline_name,
                                          followup_visit_name=followup_name, liver_mask_path=liver_mask_path,
                                          tolerance_liver=tolerance_liver)
    else:
        print("You are not incorporating anatomical information.")
    return dif_current


for patient in tqdm(sorted(patients)):

    patient_visits = list(filter(lambda x: x.startswith(patient), files))
    baseline = min(patient_visits, key=lambda visit: datetime.strptime(visit[11:19], "%Y%m%d"))
    patient_visits.remove(baseline)
    baseline_mask_pred = nib.load(os.path.join(base_pred, baseline)).get_fdata()

    dif_map_restore_pos = []
    dif_map_restore_neg = []

    for followup_visit in patient_visits:
        followup_mask_pred = nib.load(os.path.join(base_pred, followup_visit)).get_fdata()
        dif_pred = followup_mask_pred - baseline_mask_pred
        dif_pred_neg = abs(dif_pred * (dif_pred == -1))
        dif_pred_pos = dif_pred * (dif_pred == 1)
        dif_map_restore_neg.append(dif_pred_neg)
        dif_map_restore_pos.append(dif_pred_pos)

        dif_pos_livermask = refine_mask(raw_dif_map=dif_pred_pos, dif_map_restore=dif_map_restore_pos,
                                        baseline_name=baseline, followup_name=followup_visit,
                                        liver_mask_path=livermask_path, baseline_mask=baseline_mask_pred,
                                        followup_mask=followup_mask_pred)
        dif_neg_livermask = refine_mask(raw_dif_map=dif_pred_neg, dif_map_restore=dif_map_restore_neg,
                                        baseline_name=baseline, followup_name=followup_visit,
                                        liver_mask_path=livermask_path, baseline_mask=baseline_mask_pred,
                                        followup_mask=followup_mask_pred)

        new_dif_map_discrete = np.bitwise_or(dif_pos_livermask, dif_neg_livermask)
        new_dif_map_discrete[new_dif_map_discrete > 0] = 1
        dif_map = dif_pred * new_dif_map_discrete
        # if cal_dice(dif_pred, dif_map) < 1:
        #     print(f"DICE for {followup_visit} all: ", cal_dice(dif_pred, dif_map))
        #     print(f"DICE for {followup_visit} pos: ", cal_dice(dif_pred_pos, dif_pos_livermask))
        #     print(f"DICE for {followup_visit} neg: ", cal_dice(dif_pred_neg, dif_neg_livermask))
        save_img = sitk.GetImageFromArray(new_dif_map_discrete.transpose(2, 1, 0))
        save_img.SetSpacing(sitk.ReadImage(os.path.join(base_pred, followup_visit)).GetSpacing())
        save_img.SetDirection(sitk.ReadImage(os.path.join(base_pred, followup_visit)).GetDirection())
        save_img.SetOrigin(sitk.ReadImage(os.path.join(base_pred, followup_visit)).GetOrigin())
        sitk.WriteImage(save_img, os.path.join(save_refine, followup_visit))
