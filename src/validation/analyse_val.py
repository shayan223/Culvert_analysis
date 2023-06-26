import os

from ast import literal_eval
from collections import namedtuple

import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import euclidean
import seaborn as sb
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from src.config import VALIDATION_RESULTS_DIR as VAL_RES_DIR


def analyse_val():
    Rectangle = namedtuple("Rectangle", "xmin ymin xmax ymax")

    val_res = pd.read_csv(
        os.path.join(VAL_RES_DIR, "validation_results.csv"), index_col=False
    )
    labels = pd.read_csv(os.path.join(VAL_RES_DIR, "val_labels.csv"))

    labels = (
        labels.groupby(
            ["filename"],
        )
        .agg(tuple)
        .applymap(list)
        .reset_index()
    )

    def convert_to_literal(row):
        current_labels = []
        bounding_boxes = []

        label_list = literal_eval(row.iloc[0]["labels"])

        for i in label_list:
            current_labels.append(literal_eval(i))

        centroid_list = literal_eval(row.iloc[0]["centroids"])
        x1_list = literal_eval(row.iloc[0]["x1"])
        y1_list = literal_eval(row.iloc[0]["y1"])
        x2_list = literal_eval(row.iloc[0]["x2"])
        y2_list = literal_eval(row.iloc[0]["y2"])

        for i in range(len(x1_list)):
            bounding_box = Rectangle(
                x1_list[i],
                y1_list[i],
                x2_list[i],
                y2_list[i],
            )
            bounding_boxes.append(bounding_box)

        return current_labels, centroid_list, bounding_boxes

    def check_overlap(a, b):
        xA = max(a.xmin, b.xmin)
        yA = max(a.ymin, b.ymin)
        xB = min(a.xmax, b.xmax)
        yB = min(a.ymax, b.ymax)

        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0

        boxAArea = abs((a.xmax - a.xmin) * (a.ymax - a.ymin))
        boxBArea = abs((b.xmax - b.xmin) * (b.ymax - b.ymin))

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def find_centroid(a):
        x = (a.xmax + a.xmin) / 2
        y = (a.ymin + a.ymax) / 2
        return (x, y)

    def closest_center(centroids, point):
        if len(centroids) < 1:
            return -1
        closest = centroids[0]
        closest_index = 0
        for i in range(len(centroids)):
            c = centroids[i]
            if euclidean(c, point) < euclidean(closest, point):
                closest = c
                closest_index = i
        return closest_index

    y = []
    y_pred = []

    overlap_vals = []
    labels_missed = 0
    extra_labels = 0
    total_true_labels = 0

    print("Running Analysis: ")
    for i in tqdm(range(val_res.shape[0])):
        cur_label_row = labels.iloc[[i]]

        true_labels_row = cur_label_row["box_type"].iloc[0]

        label_bounds = []

        for h in range(len(cur_label_row["box_type"].iloc[0])):
            single_label_bound = Rectangle(
                cur_label_row["xmin"].iloc[0][h],
                cur_label_row["ymin"].iloc[0][h],
                cur_label_row["xmax"].iloc[0][h],
                cur_label_row["ymax"].iloc[0][h],
            )

            label_bounds.append(single_label_bound)

        label_centroids = []
        for box in label_bounds:
            label_centroids.append(find_centroid(box))

        cur_row = val_res.iloc[[i]]

        cur_preds, cur_centroids, bounding_boxes = convert_to_literal(cur_row)

        prediction_mapping = []

        for center in label_centroids:
            if not cur_centroids:
                labels_missed += len(label_centroids) - len(prediction_mapping)
                break

            closest_index = closest_center(cur_centroids, center)
            prediction_mapping.append(closest_index)
            cur_centroids.pop(closest_index)

        if len(cur_centroids) > 0:
            extra_labels += len(cur_centroids)
        total_true_labels += len(true_labels_row)

        avg_overlap = 0
        count = 0

        for index in range(len(label_bounds)):
            if index >= len(prediction_mapping):
                count += 0
                break

            if len(bounding_boxes) > 0:
                if prediction_mapping[index] >= 0:
                    overlap = check_overlap(
                        label_bounds[index],
                        bounding_boxes[prediction_mapping[index]],
                    )
                    avg_overlap += overlap
                    count += 1

        if count > 0:
            avg_overlap = avg_overlap / count
        overlap_vals.append(avg_overlap)

        if not cur_preds or (len(cur_preds) < len(label_bounds)):
            cur_preds.extend([False] * (len(label_bounds) - len(cur_preds)))

        elif len(cur_preds) > len(true_labels_row):
            revised_preds = []
            for pred_loc in range(len(prediction_mapping)):
                revised_preds.append(cur_preds[prediction_mapping[pred_loc]])
            cur_preds = revised_preds

        y.extend(true_labels_row)
        y_pred.extend(cur_preds)

    print("Done!")

    avg_overlap = sum(overlap_vals) / val_res.shape[0]
    label_miss_rate = labels_missed / total_true_labels
    extra_label_rate = extra_labels / total_true_labels

    conf_mat = confusion_matrix(y, y_pred)
    print(conf_mat)

    accuracy = conf_mat.diagonal().sum() / conf_mat.sum()
    print("Accuracy: ", accuracy)

    ax = plt.subplot()

    alt_title = f"Accuracy: {accuracy}, IOU: {avg_overlap}\n"
    alt_title += f"Miss: {label_miss_rate}, Extra: {extra_label_rate}"

    sb.heatmap(conf_mat, annot=True, fmt="d").set(title=alt_title)

    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")

    plt.savefig(os.path.join(VAL_RES_DIR, "confusion_matrix.jpg"))
    plt.clf()