from matplotlib import pyplot as plt
import numpy as np
import sys
import tensorflow as tf
import matplotlib
from PIL import Image
import matplotlib.patches as patches
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import argparse
import os
import streamlit as st

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

@st.cache
def prepare_image(image_path):

    model_path = './trained_model/frozen_inference_graph.pb'

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v2.io.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


    label_map = label_map_util.load_labelmap('./trained_model/labels.txt')
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            # parser = argparse.ArgumentParser()
            # parser.add_argument('image_path')
            # args = parser.parse_args()
            # image_path = os.path.join("images", "5.jpg")
            image_np = load_image_into_numpy_array(Image.open(image_path))
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: np.expand_dims(image_np, axis=0)})

            if scores[0][0] < 0.1:
                sys.exit('Wally not found :(')
            # names = [[str(n)] for n in range(300)]
    return image_np, boxes, scores


def set_values(total, box, score):
    width = total.shape[0]
    height = total.shape[1]
    total[int(np.round(box[0]*width,0)):int(np.round(box[2]*width,0)), 
                int(np.round(box[1]*height,0)):int(np.round(box[3]*height,0))] += score
    return total

image_n = st.selectbox("Which image do you want to see?",[str(n) for n in range(1,len(os.listdir("images"))-1)])

image_np, boxes, scores = prepare_image(os.path.join("images", f"{image_n}.jpg"))

total = np.zeros((image_np.shape[0], image_np.shape[1]))
for i, box in enumerate(boxes[0]):
    # print(box)
    # shades = np.zeros((image_np.shape[0], image_np.shape[1]))
    score = scores[0][i]
    # if score > 0.05:
    #     score = 0.0
    total = set_values(total, box, score)

alpha_val = st.slider("Blur image", 0, 100, 0, 1)

# import plotly.express as px

# median = np.median(total)
alphas = -np.log(total+0.000000000000001)
alphas_scaled = (alphas - np.min(alphas))/np.ptp(alphas) / 100

fig = plt.figure()
plt.imshow(image_np)
plt.imshow(np.ones((image_np.shape[0], image_np.shape[1])),  "Greys", alpha=alphas_scaled*alpha_val)
st.pyplot(fig)

np.nonzero(total)



# vis_util.draw_bounding_boxes_on_image_array(
# image_np,
# np.squeeze(boxes),
# display_str_list_list=names,
# thickness=8,
# color='green')
# plt.figure(figsize=(12, 8))
# plt.imshow(image_np)
# plt.show()



# boxes_round = np.around(boxes, decimals=2)
# print(boxes_round)



    # print(len(scores[0]))
    # print('Wally found')
    # vis_util.visualize_boxes_and_labels_on_image_array(
    #     image_np,
    #     np.squeeze(boxes),
    #     np.squeeze(classes).astype(np.int32),
    #     np.squeeze(scores),
    #     category_index,
    #     use_normalized_coordinates=True,
    #     line_thickness=8)
    # vis_util.draw_bounding_boxes_on_image_array(
    # image_np,
    # np.squeeze(boxes),
    # display_str_list_list=names,
    # thickness=8,
    # color='green')
    # plt.figure(figsize=(12, 8))
    # plt.imshow(image_np)
    # plt.show()
