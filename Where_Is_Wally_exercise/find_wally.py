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

####################
# Define functions #
####################

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

@st.cache
def set_values(total: np.array, box: np.array, score: float):
    """takes a box indicating position on image and an associated score.
    Adds to the total array the score of the box at the same positions 
    Returns the total array with new values inserted

    Args:
        total (np.array): array storing sum of scores of relevant boxes
        box (np.array): current box coordinates (xmin, ymin, xmax, ymax)
        score (float): value between 0 and 1 indicating certainty

    Returns:
        total (np.array): the updated array
    """    
    width = total.shape[0]
    height = total.shape[1]
    total[int(np.round(box[0]*width,0)):int(np.round(box[2]*width,0)), 
                int(np.round(box[1]*height,0)):int(np.round(box[3]*height,0))] += score
    return total

def get_mid_point(width, height):
    return width/2, height/2

@st.cache
def plot_pic_dot(image, dot_x, dot_y, alphas, scale_val):
    fig = plt.figure()
    plt.imshow(image)
    plt.imshow(np.ones((image.shape[0], image.shape[1])),  "Greys", alpha=alphas*scale_val)
    plt.plot(dot_x, dot_y, 'bo')
    return fig


if __name__ == "__main__":
    st.header("Find Wally in the picture!")
    st.write("You can blur the image by increasing the slider value.\
             Move the dot around by pressing the button, and confirm when it points to Wally")
    image_n = st.selectbox("Which image do you want to see?",[str(n) for n in range(1,len(os.listdir("images")))])

    image_np, boxes, scores = prepare_image(os.path.join("images", f"{image_n}.jpg"))

    total = np.zeros((image_np.shape[0], image_np.shape[1]))
    for i, box in enumerate(boxes[0]):
        score = scores[0][i]
        total = set_values(total, box, score)

    # Prepare blurring
    alpha_val = st.slider("Increase blurring of low probability areas", 0, 100, 0, 1)
    alphas = -np.log(total+0.000000000000001)
    alphas_scaled = (alphas - np.min(alphas))/np.ptp(alphas) / 100

    # Prepare initial point
    x_val, y_val = get_mid_point(image_np.shape[0], image_np.shape[1])

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        x_up = st.button("Increase x position")
    with c2:
        x_down = st.button("Decrease x position")
    with c3:
        y_up = st.button("Increase y position")
    with c4:
        y_down = st.button("Decrease y position")

    if x_up:
        x_val += 50
    if x_down:
        x_val -=  50
    if y_up:
        y_val -=  50
    if y_down:
        y_val += 50

    fig = plot_pic_dot(image_np, x_val, y_val, alphas_scaled, alpha_val)
    st.pyplot(fig)

    if st.button("I found Wally") or st.button("I dont think Wally is there"):
        st.write("Thanks a lot")


    # t = np.arange(0.0, 1.0, 0.01)
    # s = np.sin(2 * np.pi * t)
    # fig, ax = plt.subplots()
    # fig.canvas.mpl_connect('button_press_event', onclick)
    # plt.show()
    # st.pyplot(fig)



    # binding_id = plt.connect('motion_notify_event', on_move)
    # plt.connect('button_press_event', on_click)

    





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
