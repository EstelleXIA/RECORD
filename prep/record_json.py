import os
import json
from datetime import datetime
import argparse
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument('--input_img', help='Input folder for serial CT images', required=True)
parser.add_argument('--input_mask', help='Input folder for serial segmentation model predicted masks', required=True)
parser.add_argument('--output', help='Output folder for generated .json file', required=True)
parser.add_argument('--test_id', help='testing patients', required=True)

args = parser.parse_args()
input_img_path = args.input_img
input_mask_path = args.input_mask
save_path = args.output

files = sorted(os.listdir(input_img_path))
patients = list(set(list(map(lambda x: x[:10], files))))
test_id = args.test_id
with open("/dssg/home/acct-clsyzs/clsyzs-beigene/BTCV/submit/prep/test.txt", "r") as f:
    test_patients = [x.split("\n")[0] for x in f.readlines()]
train_patients = deepcopy(patients)
for test in test_patients:
    train_patients.remove(test)

print(f"There are {len(train_patients)} training samples and {len(test_patients)} validation samples.\n"
      f"Some training samples are {train_patients[:5] if len(train_patients) > 5 else train_patients}. \n"
      f"Some validation samples are {test_patients[:5] if len(test_patients) > 5 else test_patients}. \n")

pair_json = {"training": [], "validation": []}
for patient in patients:
    patient_visits = list(filter(lambda x: x.startswith(patient), files))
    baseline = min(patient_visits, key=lambda visit: datetime.strptime(visit[11:19], "%Y%m%d"))
    patient_visits.remove(baseline)
    key = "training" if patient in train_patients else "validation"
    pair_json[key].extend(
        [{"image": os.path.join(input_img_path, baseline),
          "label": os.path.join(input_mask_path, baseline.replace("_0000", "")),
          "image_1": os.path.join(input_img_path, followup),
          "label_1": os.path.join(input_mask_path, followup.replace("_0000", ""))}
         for followup in patient_visits])
    with open(os.path.join(save_path, "ants_json.json"), "w") as f:
        json.dump(pair_json, f)
