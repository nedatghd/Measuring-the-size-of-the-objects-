# -*- coding: utf-8 -*-
"""Measurement.ipynb



!pip install -U --pre tensorflow


# Commented out IPython magic to ensure Python compatibility.
# # Install the Object Detection API
# 
%%bash
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .

!pip install -U numpy

"""
import PIL
from PIL import Image
from scipy.spatial.distance import euclidean
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import imutils
import cv2
from google.colab.patches import cv2_imshow
import matplotlib
import matplotlib.pyplot as plt
import pathlib

import os
import shutil
import random
import io
import imageio
import glob
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Javascript
from IPython.display import Image as IPyImage

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import colab_utils
from object_detection.builders import model_builder
# tf.config.list_physical_devices('GPU')

# %matplotlib inline

"""Utilities"""

def remove_folder_contents(path):
    shutil.rmtree(path)
    os.makedirs(path)

def remove_files_in_folder(folderPath):
        # loop through all the contents of folder
        for filename in os.listdir(folderPath):
            # remove the file
            os.remove(f"{folderPath}/{filename}")

def midpoint(ptA, ptB):
  return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def scale_calibration(num_pixel):
  pixel_per_metric = 24.5 / 166
  return num_pixel*pixel_per_metric

def k_means_segmentation(pil_image, im_show=False):

    # Convert to HSV format
    opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    img_rgb = cv2.cvtColor(opencvImage,cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2HSV)

    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = img.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)
    # print(pixel_values.shape)

    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # number of clusters (K)
    k = 2
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(img.shape)

    if im_show:
      # show the image
      plt.imshow(segmented_image)
      plt.show()
    return segmented_image, opencvImage

def apply_morphology(segmented_image, kernel, im_show=False):

    #Apply preprocessing to remove noise
    grayscale = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)   #converting the color to grayscale

    opening = cv2.morphologyEx(grayscale, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    # show the image
    if im_show:
        plt.imshow(closing)
        plt.show()
    return closing

def show_data_point_of_curve_polygon(img_rgb, closing_img, im_show = False):

    #  Python code to find the co-ordinates of the contours detected in an image.
    imgrgb = img_rgb.copy()
    # Reading image
    font = cv2.FONT_HERSHEY_COMPLEX

    # Converting image to a binary image
    # ( black and white only image).
    _, thresh = cv2.threshold(closing_img.copy(), 110, 255, cv2.THRESH_BINARY)
    # plt.imshow(threshold)
    # Detecting contours in image.
    contours, _= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Going through every contours found in the image.
    for cnt in contours :

        area = cv2.contourArea(cnt)
        if area > 1000:


            approx = cv2.approxPolyDP(cnt, 0.010 * cv2.arcLength(cnt, True), True)

            # draws boundary of contours.
            cv2.drawContours(imgrgb, [approx], 0, (0, 0, 255), 1)

            # Used to flatted the array containing
            # the co-ordinates of the vertices.
            n = approx.ravel()
            i = 0

            for j in n :
                if(i % 2 == 0):
                    x = n[i]
                    y = n[i + 1]

                    # String containing the co-ordinates.
                    string = str(x) + " " + str(y)

                    if(i == 0):
                        # text on topmost co-ordinate.
                        cv2.putText(imgrgb, "Arrow tip", (int(x), int(y)),font, 0.4, (255, 0, 0))
                    else:
                        # text on remaining co-ordinates.
                        cv2.putText(imgrgb, string, (int(x), int(y)),
                                  font, 0.4, (0, 0, 255))
                i = i + 1

    # Showing the final image.
    if im_show :
        plt.imshow(imgrgb)
        plt.show()
    return imgrgb, thresh


def show_dimentionality_polygon(img_rgb, closing_img, im_show = False):


    # Reading image
    font = cv2.FONT_HERSHEY_COMPLEX
    imgrgb = img_rgb.copy()
    # imgrgb2 = img_rgb.copy()
    ret, thresh = cv2.threshold(closing_img.copy(), 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by area
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    n_detected_c = len(sorted_contours)
    # print("number of detected contours :", n_detected_c)

    # number of desired contours
    n_desired_c = 2

    # comparison
    if n_detected_c >= n_desired_c:
      up_to = n_desired_c
    else :
      up_to = n_detected_c

    k = 0
    for c in sorted_contours[:up_to]:


                perimeter = cv2.arcLength(c, True)
                perimeter = round(perimeter, 4)
                area = cv2.contourArea(c)
                # print('Area:', area)
                # print('Perimeter:', perimeter)

                # x1, y1 = c[0,0]
                # cv2.putText(imgrgb, f'Area:{area}', (x1-100, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                # cv2.putText(imgrgb, f'Perimeter:{perimeter}', (x1-100, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                # cv2.putText(imgrgb, "original contour, num_pts={}".format(len(c)), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                #----------------------------------------------------
                # Draw a Blue rectangle shaped boundiong box around circle
                #----------------------------------------------------

                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(imgrgb, (x, y), (x + w, y + h), (255, 0,0), 1)

                if k == 0:
                    cv2.putText(imgrgb, f'width of rectangle :{round(scale_calibration(w), 1)} mm', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.putText(imgrgb, f'hiegth of rectangle :{round(scale_calibration(h),1)} mm', (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                #----------------------------------------------------
                # Draw a Yellow outline around the contour
                #----------------------------------------------------
                cv2.drawContours(imgrgb, [c], -1, (0,255,255), 2)

                #----------------------------------------------------
                # calculate center and radius of minimum enclosing circle
                # and draw circle pink shaped bounding box around circle
                #----------------------------------------------------
                (x_center,y_center) ,radius = cv2.minEnclosingCircle(c)
                # cast to integers
                center = (int(x_center),int(y_center))
                radius = int(radius)
                cv2.circle(imgrgb, center, radius, (255, 0, 255), 1)

                if k == 0 :
                    cv2.putText(imgrgb, f'Diameter of bigger circle :{round(scale_calibration(radius*2), 1)} mm', (center[0]-40, center[1]-200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.putText(imgrgb, f'Radius of bigger circle :{round(scale_calibration(radius), 1)} mm', (center[0]-40, center[1]-180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    #----------------------------------------------------
                    # visuallize_data_point_of_curve_polygon and mesure the dimention
                    #----------------------------------------------------
                    # approx = cv2.approxPolyDP(c, 0.015 * cv2.arcLength(c, True), True)

                    # # draws boundary of contours.
                    # cv2.drawContours(imgrgb2, [approx], 0, (0, 0, 255), 1)

                    # # Used to flatted the array containing
                    # # the co-ordinates of the vertices.
                    # n = approx.ravel()
                    # i = 0

                    # for j in n :
                    #     if(i % 2 == 0):
                    #         x = n[i]
                    #         y = n[i + 1]

                    #         # String containing the co-ordinates.
                    #         string = str(x) + " " + str(y)

                    #         if(i == 0):
                    #             # text on topmost co-ordinate.
                    #             cv2.putText(imgrgb2, "Arrow tip", (int(x), int(y)),font, 0.4, (255, 0, 0))
                    #         else:
                    #             # text on remaining co-ordinates.
                    #             cv2.putText(imgrgb2, string, (int(x), int(y)),
                    #                       font, 0.4, (0, 0, 255))
                    #     i = i + 1


                else :
                    cv2.putText(imgrgb, f'Diameter:{round(scale_calibration(radius*2), 1)} mm', (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.putText(imgrgb, f'Radius:{round(scale_calibration(radius), 1)} mm', (center[0], center[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                k += 1


    # Showing the final image.
    if im_show :
        plt.imshow(imgrgb)
        plt.show()
    return imgrgb, thresh

def show_dimentionality_rectangle(img_rgb, closing_img, im_show = False):


    imgrgb = img_rgb.copy()
    # Reading image
    font = cv2.FONT_HERSHEY_COMPLEX

    ret, thresh = cv2.threshold(closing_img.copy(), 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort the contours by area
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cordinate = []
    for cnt in sorted_contours :

        area = cv2.contourArea(cnt)
        if area >= cv2.contourArea(sorted_contours[0]):

            perimeter = cv2.arcLength(cnt, True)
            perimeter = round(perimeter, 2)

            #----------------------------------------------------
            # Draw a Yellow outline around the contour
            #----------------------------------------------------
            cv2.drawContours(imgrgb, [cnt], -1, (0,255,255), 2)

            #----------------------------------------------------
            # Draw a Blue rectangle shaped boundiong box around object
            #----------------------------------------------------
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(imgrgb, (x, y), (x + w, y + h), (255, 0,0), 1)

            #----------------------------------------------------
            # Draw a Red rectangle boundiong box generated from approximated data point of the curve
            #----------------------------------------------------
            approx = cv2.approxPolyDP(cnt, 0.002 * cv2.arcLength(cnt, True), True)

            # draws boundary of contours.
            cv2.drawContours(imgrgb, [approx], 0, (0, 0, 255), 1)

            # Used to flatted the array containing
            # the co-ordinates of the vertices.
            n = approx.ravel()
            i = 0

            for j in n :
                if(i % 2 == 0):
                    x = n[i]
                    y = n[i + 1]

                    # String containing the co-ordinates.
                    string = str(x) + " " + str(y)
                    cordinate.append((x,y))

                    # print("approximated data point of the curve in term of pixel:", x,y,"x,y")
                    if(i == 0):
                        # text on topmost co-ordinate.
                        cv2.putText(imgrgb, string, (int(x-10), int(y-10)),font, 0.4, (0, 0, 255))
                    else:
                        # text on remaining co-ordinates.
                        cv2.putText(imgrgb, string, (int(x-10), int(y-10)),
                                  font, 0.4, (0, 0, 255))
                i = i + 1

    # Showing the final image.

    center = midpoint(midpoint(cordinate[0], cordinate[1]), midpoint(cordinate[2], cordinate[3]))
    # compute the Euclidean distance between the midpoints
    dA = dist.euclidean(cordinate[0], cordinate[1])
    cv2.putText(imgrgb, f'hiegth:{round(scale_calibration(dA), 1)}', (int(center[0]),int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    dB = dist.euclidean(cordinate[1], cordinate[2])
    cv2.putText(imgrgb, f'width:{round(scale_calibration(dB),1)}', (int(center[0]),int(center[1])+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    x1, y1 = cnt[0,0]  # to determine the coordinates
    area_of_rectangle = round(scale_calibration(dB),2) * round(scale_calibration(dA), 2)
    cv2.putText(imgrgb, f'Area:{area_of_rectangle} mm^2', (x1+int(center[0]), y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    perimeter_of_rectangle = (round(scale_calibration(dB),2) + round(scale_calibration(dA), 2)) * 2
    cv2.putText(imgrgb, f'Perimeter:{perimeter} mm', (x1+int(center[0]), y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2_imshow(imgrgb)

    # Showing the final image.
    if im_show :
        plt.imshow(imgrgb)
        plt.show()
    return imgrgb, thresh


def show_dimentionality_circle(img_rgb, closing_img, im_show = False):

    # Reading image
    font = cv2.FONT_HERSHEY_COMPLEX
    imgrgb = img_rgb.copy()
    ret, thresh = cv2.threshold(closing_img.copy(), 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by area
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    n_detected_c = len(sorted_contours)
    # print("number of detected contours :", n_detected_c)

    # number of desired contours
    n_desired_c = 7

    # comparison
    if n_detected_c >= n_desired_c:
      up_to = n_desired_c
    else :
      up_to = n_detected_c
    i = 0
    for c in sorted_contours[:up_to]:


                perimeter = cv2.arcLength(c, True)
                perimeter = round(perimeter, 4)
                area = cv2.contourArea(c)
                # print('Area:', area)
                # print('Perimeter:', perimeter)

                # x1, y1 = c[0,0]
                # cv2.putText(imgrgb, f'Area:{area}', (x1-100, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                # cv2.putText(imgrgb, f'Perimeter:{perimeter}', (x1-100, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                # cv2.putText(imgrgb, "original contour, num_pts={}".format(len(c)), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                #----------------------------------------------------
                # Draw a Blue rectangle shaped boundiong box around circle
                #----------------------------------------------------

                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(imgrgb, (x, y), (x + w, y + h), (255, 0,0), 1)

                #----------------------------------------------------
                # Draw a Yellow outline around the contour
                #----------------------------------------------------
                cv2.drawContours(imgrgb, [c], -1, (0,255,255), 2)

                #----------------------------------------------------
                # calculate center and radius of minimum enclosing circle
                # and draw circle pink shaped bounding box around circle
                #----------------------------------------------------
                (x_center,y_center) ,radius = cv2.minEnclosingCircle(c)
                # cast to integers
                center = (int(x_center),int(y_center))
                radius = int(radius)
                cv2.circle(imgrgb, center, radius, (255, 0, 255), 1)

                if i == 0 :
                    cv2.putText(imgrgb, f'Diameter of bigger circle :{round(scale_calibration(radius*2), 1)} mm', (center[0]-40, center[1]-200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.putText(imgrgb, f'Radius of bigger circle :{round(scale_calibration(radius), 1)} mm', (center[0]-40, center[1]-180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                else :
                    cv2.putText(imgrgb, f'Diameter:{round(scale_calibration(radius*2), 1)} mm', (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                    cv2.putText(imgrgb, f'Radius:{round(scale_calibration(radius), 1)} mm', (center[0], center[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

                i += 1


    # Showing the final image.
    if im_show :
        plt.imshow(imgrgb)
        plt.show()
    return imgrgb, thresh


def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path.

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name=None):
  """Wrapper function to visualize detections.

  Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    figsize: size for the figure.
    image_name: a name for the image file.
  """
  image_np_with_annotations = image_np.copy()
  viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_annotations,
      boxes,
      classes,
      scores,
      category_index,
      use_normalized_coordinates=True,
      min_score_thresh=0.8)
  if image_name:
    plt.imsave(image_name, image_np_with_annotations)
  else:
    plt.imshow(image_np_with_annotations)


def get_model_detection_function(model):
  """Get a tf.function for detection."""

  @tf.function
  def detect_fn(image):
    """Detect objects in image."""

    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

  return detect_fn
