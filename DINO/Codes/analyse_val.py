from cProfile import label
from queue import Empty
import pandas as pd
from sklearn.metrics import confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sb
from ast import literal_eval
from collections import namedtuple
from scipy.spatial.distance import euclidean
from tqdm import tqdm



def analyse_val():
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

    # val_res = pd.read_csv('./validation_results/validation_results.csv',index_col=False)
    # labels = pd.read_csv('./validation_results/val_labels.csv')
    val_res = pd.read_csv('/mnt/sdb1/udaykanth/DETR/validation_results/validation_results.csv',index_col=False)
    labels = pd.read_csv('/mnt/sdb1/udaykanth/DETR/validation_results/val_labels.csv')


    labels = labels.groupby(['filename']).agg(tuple).applymap(list).reset_index()

    # Data frame stores nested data types as string, so we want to convert them back to usable data types
    def convert_to_literal(row):
        current_labels = []
        bounding_boxes = []
        
        label_list = literal_eval(row.iloc[0]['labels'])

        for i in label_list:
            current_labels.append(literal_eval(i))

        centroid_list = literal_eval(row.iloc[0]['centroids'])
        x1_list = literal_eval(row.iloc[0]['x1'])    
        y1_list = literal_eval(row.iloc[0]['y1'])
        x2_list = literal_eval(row.iloc[0]['x2'])
        y2_list = literal_eval(row.iloc[0]['y2'])

        #All points should come in pairs, so we can just use the length of the first one
        #We construct bounding boxes with all their respective points
        for i in range(len(x1_list)):
            bounding_box = Rectangle(x1_list[i],y1_list[i],x2_list[i],y2_list[i])
            bounding_boxes.append(bounding_box)

        return current_labels, centroid_list, bounding_boxes

    # returns 0 if rectangles don't intersect
    def check_overlap(a, b): 
        ''' 
        dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
        dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
        print(dx*dy)
        if (dx>=0) and (dy>=0):
            return dx*dy
        else:
            return 0
        '''

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(a.xmin, b.xmin) #boxA[0] boxB[0]
        yA = max(a.ymin, b.ymin) #boxA[1] boxB[1]
        xB = min(a.xmax, b.xmax) #boxA[2] boxB[2]
        yB = min(a.ymax, b.ymax) #boxA[3] boxB[3]

        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = abs((a.xmax - a.xmin) * (a.ymax - a.ymin))
        boxBArea = abs((b.xmax - b.xmin) * (b.ymax - b.ymin))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def find_centroid(a):  
        x = (a.xmax + a.xmin) / 2
        y = (a.ymin + a.ymax) / 2
        return (x,y)

    def closest_center(centroids, point):
        #return with error value if centroid list is empty
        if(len(centroids) < 1): 
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
    y_pred = []#labels.box_type.tolist()

    #Only counting boxes which have a corresponding label
    #(not counting extra bounds produced by the model)
    overlap_vals = []
    #Count the number of times a model didn't label enough
    labels_missed = 0
    #Count the number of times a model made too many labels
    extra_labels = 0
    total_true_labels = 0
    '''
    for i in val_res['labels']:
        #We use literal eval to convert string interpratations of lists back to python list objects
        item = literal_eval(i)
        #Loop through all lists of labels
        if(item):
            y.append(literal_eval(item[0])) # if the label exists, take the first one
        else:
            y.append(False)# Otherwise append a false label

    '''
    print("Running Analysis: ")
    for i in tqdm(range(val_res.shape[0])):#val_res['labels']:
        cur_label_row = labels.iloc[[i]]
        
        #true_labels_row = [] 
        #true_labels_row.append(cur_label_row['box_type'].iloc[0])
        true_labels_row = cur_label_row['box_type'].iloc[0]

        label_bounds = []
        # generate bounding boxes from ground truth data

        for h in range(len(cur_label_row['box_type'].iloc[0])):
            # TODO not currently using list, so this needs to be changed as well
            single_label_bound = Rectangle(cur_label_row['xmin'].iloc[0][h],cur_label_row['ymin'].iloc[0][h],
                                    cur_label_row['xmax'].iloc[0][h],cur_label_row['ymax'].iloc[0][h])

            label_bounds.append(single_label_bound)

        label_centroids = []
        for box in label_bounds:
            label_centroids.append(find_centroid(box))

        #We use literal eval to convert string interpratations of lists back to python list objects
        cur_row = val_res.iloc[[i]]

        cur_preds, cur_centroids, bounding_boxes = convert_to_literal(cur_row)

        # assign predicted bounding boxes to their closest label, only one per label. 

        #contains indeces mapping from label to prediction boxes
        #ex: prediction_mapping[4] = 6 means the 5th label centroid is mapped to the 7th predicted bounding box's centroid
        prediction_mapping = []

        for center in label_centroids:
            #If the loop has to end early because of a lack of predictions
            #then we keep track of how many the model missed
            if not cur_centroids:
                labels_missed += (len(label_centroids) - len(prediction_mapping))
                break

            #closest = min(cur_centroids, key=lambda c : euclidean(c, center))
            closest_index = closest_center(cur_centroids,center)
            prediction_mapping.append(closest_index)
            #remove the centroid from consideration once we've used it to map the respective bounding box
            cur_centroids.pop(closest_index)
        
        #count extra unused bounds from the model
        if(len(cur_centroids) > 0):
            extra_labels += len(cur_centroids)
        total_true_labels += len(true_labels_row)

        avg_overlap = 0
        count = 0

        for index in range(len(label_bounds)):
            #if there aren't enough predicted boxes for labels, count as a single 0 overlap box
            if(index >= len(prediction_mapping)):
                # TODO test with no-op counting as 0 overlap, and with it not 
                #count += 1
                count += 0
                break
            #We only want to compare against boxes the model actually attempted
            if(len(bounding_boxes) > 0):
                if(prediction_mapping[index] >= 0):

                    overlap = check_overlap(label_bounds[index],bounding_boxes[prediction_mapping[index]])
                    #Add to the total to compute average overlap
                    avg_overlap += overlap
                    count += 1

        if(count > 0):
            avg_overlap = avg_overlap / count
        overlap_vals.append(avg_overlap)

        
        #for the sake of a confusion matrix, we count no-ops as False labels

        if not cur_preds or (len(cur_preds) < len(label_bounds)):
            #Fill in with false labels until the length of the predictions matches labels
            cur_preds.extend([False] * (len(label_bounds) - len(cur_preds)))
            #for j in range(len(label_class)):
                #cur_preds.extend(['False'] * (len(label_centroids) - len(prediction_mapping)))
                #print((len(label_centroids) - len(prediction_mapping)))

        #trim off extra predictions for the same reason
        elif len(cur_preds) > len(true_labels_row):
            revised_preds = []
            for pred_loc in range(len(prediction_mapping)):
                revised_preds.append(cur_preds[prediction_mapping[pred_loc]])
            cur_preds = revised_preds

        y.extend(true_labels_row)
        y_pred.extend(cur_preds)



        
    print("Done!")

    #all averages here are per image (hence the division by number of all images)
    #print(overlap_vals)
    #print(len(overlap_vals))
    #print(len(y))
    #print(len(y_pred))
    avg_overlap = sum(overlap_vals) / val_res.shape[0]
    label_miss_rate = labels_missed / total_true_labels
    extra_label_rate = extra_labels / total_true_labels

    conf_mat = confusion_matrix(y,y_pred)
    print(conf_mat)

    #Use the first accuracy for per-class accuracy
    #accuracy = conf_mat.diagonal()/conf_mat.sum(axis=1)
    accuracy = conf_mat.diagonal().sum()/conf_mat.sum()
    print('Accuracy: ', accuracy)

    ax = plt.subplot()
    alt_title = 'Accuracy: '+ str(accuracy) + ', IOU: ' + str(avg_overlap) + \
        ', \n Miss: ' + str(label_miss_rate) + ', Extra: ' + str(extra_label_rate)

    sb.heatmap(conf_mat,annot=True,fmt='d').set(title=alt_title)
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    plt.savefig('./validation_results/confusion_matrix_rcnn_res.png')
    plt.savefig('/mnt/sdb1/udaykanth/DETR/validation_results/confusion_matrix_rcnn_res.png')
    plt.clf()

