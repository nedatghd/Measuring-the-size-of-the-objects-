#!/usr/bin/python

# # USAGE
# python run.py \
# --image_dir TEST/images \
# --pipelinefile new_config/pipeline.config \
# --cheakpoint  new_config/cheakpoint

from utils import *
import pathlib
from object_detection.utils import config_util
from object_detection.builders import model_builder

"""#recover our saved Tensorflow model"""
"""load the saved model weight"""


# construct the argument parse and parse the arguments
import argparse
from argparse import ArgumentParser
ap = ArgumentParser()

ap.add_argument("--pipelinefile", type=str, required=True,
	help="pipelinefile for object detection")

ap.add_argument("--cheakpoint", type=str, required=True,
	help="cheakpoint for object detection")

ap.add_argument("--rawdir", 
  help="Directory path to raw images.", default="./data/raw",
  type=str)

ap.add_argument("--save_tf_detection", action="store_false",
                    help="whether to save tf output or no")

ap.add_argument("--savedir_resized", 
  help="Directory path to save resized images.", default="./data/images",
  type=str)

ap.add_argument("--cropdir", 
  help="Directory path to cropped images after feeding to tf object detection.", default="./data/images",
  type=str)

ap.add_argument("--resultdir", 
  help="Directory path to result images after measurement.", default="./data/images",
  type=str)

ap.add_argument("--ext", 
  help="Raw image files extension to resize.", default="jpg", type=str)

ap.add_argument("--targetsize", 
  help="Target size to resize as a tuple of 2 integers.", default="(800, 600)",
  type=str)

if __name__ == "__main__":

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    print(tf.config.list_physical_devices('GPU'))
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    print(tf.test.is_built_with_cuda())

    args = vars(ap.parse_args())
    raw_dir = args["rawdir"]
    save_dir_resized = args["savedir_resized"]
    save_dir_cropped = args["cropdir"]
    save_dir_result = args["resultdir"]
    ext = args["ext"]
    target_size = eval(args["targetsize"])
    msg = "--target-size must be a tuple of 2 integers"
    assert isinstance(target_size, tuple) and len(target_size) == 2, msg
    fnames = glob.glob(os.path.join(raw_dir, "*.{}".format(ext)))

    if os.path.exists(save_dir_resized):
       remove_files_in_folder(save_dir_resized)
    else:
       os.makedirs(save_dir_resized, exist_ok=True)

    if os.path.exists(save_dir_cropped):
       remove_files_in_folder(save_dir_cropped)
    else:
       os.makedirs(save_dir_cropped, exist_ok=True)

    if os.path.exists(save_dir_result):
       remove_files_in_folder(save_dir_result)
    else:
       os.makedirs(save_dir_result, exist_ok=True)

    # print(
    #     "{} files to resize from directory `{}` to target size:{}".format(
    #         len(fnames), raw_dir, target_size
    #     )
    # )
    for i, fname in enumerate(fnames):
        print(".", end="", flush=True)
        img = cv2.imread(fname)
        img_small = cv2.resize(img, target_size)
        new_fname = "{}.{}".format(str(i), ext)
        small_fname = os.path.join(save_dir_resized, new_fname)
        cv2.imwrite(small_fname, img_small)
    # print(
    #     "\nDone resizing {} files.\nSaved to directory: `{}`".format(
    #         len(fnames), save_dir_resized
    #     )
    # )



    filenames_ckpt = list(pathlib.Path(args["cheakpoint"]).glob('*.index'))
    pipeline_file = args["pipelinefile"]

    # filenames_ckpt.sort()
    # print(filenames_ckpt)

    pipeline_config = pipeline_file
    #generally you want to put the last ckpt from training in here
    model_dir = str(filenames_ckpt[-1]).replace('.index','')
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    # print("configs", configs)
    model_config = configs['model']
    detection_model = model_builder.build(
          model_config=model_config, is_training=False)

    # Restore checkpoint

    ckpt = tf.compat.v2.train.Checkpoint(
          model=detection_model)
    ckpt.restore(os.path.join(str(filenames_ckpt[-1]).replace('.index','')))


    detect_fn = get_model_detection_function(detection_model)

    label_id_offset =1

    circle_CLASS_ID = 1
    polygon_CLASS_ID = 2
    rectangle_CLASS_ID = 3

    category_index = {circle_CLASS_ID :
                        {'id'  : circle_CLASS_ID,'name': 'circle'},
                      polygon_CLASS_ID :
                          {'id'  : polygon_CLASS_ID,'name': 'polygon'},
                      rectangle_CLASS_ID :
                          {'id'  : rectangle_CLASS_ID,'name': 'rectangle'}
                      }

    # we inputted the resized image into the model ...

    img_files = [x for x in os.listdir(save_dir_resized) if x.endswith(('jpg', 'png'))]
    test_images_np = []


    for images in img_files:

        image_dir = os.path.join(save_dir_resized, images)
        image = Image.open(image_dir)
        # image.show()

        test_images_np.append(np.expand_dims(load_image_into_numpy_array(image_dir), axis=0))

        np_image = np.array(image)
        (frame_height, frame_width) = np_image.shape[:2]

        input_tensor = tf.convert_to_tensor(np.expand_dims(np_image, 0), dtype=tf.float32)
        # print("The resized shape of the tensor is :", input_tensor.shape)


        detections, predictions_dict, shapes = detect_fn(input_tensor)

        index_bbx = np.argmax(np.array(detections["detection_scores"][0]))
        y_min, x_min, y_max, x_max = detections["detection_boxes"][0][index_bbx]

        my_classes = detections['detection_classes'][0].numpy() + label_id_offset
        my_scores = detections['detection_scores'][0].numpy()

        min_score = 0.5

        object_name = [category_index[value]['name'] for index,value in enumerate(my_classes)  if my_scores[index] > min_score]
        # print("The object is :", object_name[0])

        ymin = int((np.array(y_min)*frame_height))
        xmin = int((np.array(x_min)*frame_width))
        ymax = int((np.array(y_max)*frame_height))
        xmax = int((np.array(x_max)*frame_width))

        # Perform the cropping operation
        cropped_img = np_image[ymin:ymax,xmin:xmax]
        cropped_img = Image.fromarray(cropped_img)
        width, height = cropped_img.size

        # cropped_img.show()
        cropped_img.save(os.path.join(save_dir_cropped, f"cropped_{str(images)}"))

        if object_name[0] == "polygon":
          segmented_cropped_image, opencvImage = k_means_segmentation(cropped_img,im_show=False)
          cv2.imwrite(os.path.join(save_dir_result, f"segmented_cropped_image_{str(images)}"), segmented_cropped_image)
          closing_img = apply_morphology(segmented_cropped_image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)),im_show=False)
          cv2.imwrite(os.path.join(save_dir_result, f"morphology_{str(images)}"), closing_img)
          out, thresh = show_dimentionality_polygon(opencvImage, closing_img, im_show=False)
          cv2.imwrite(os.path.join(save_dir_result, f"binarized_{str(images)}"), thresh)
          cv2.imwrite(os.path.join(save_dir_result, f"measured_{str(images)}"), out)


        if object_name[0] == "rectangle":
          segmented_cropped_image, opencvImage = k_means_segmentation(cropped_img,im_show=False)
          cv2.imwrite(os.path.join(save_dir_result, f"segmented_cropped_image_{str(images)}"), segmented_cropped_image)
          closing_img = apply_morphology(segmented_cropped_image, cv2.getStructuringElement(cv2.MORPH_RECT,(17,17)),im_show=False)
          cv2.imwrite(os.path.join(save_dir_result, f"morphology_{str(images)}"), closing_img)
          out, thresh = show_dimentionality_rectangle(opencvImage, closing_img, im_show=False)
          cv2.imwrite(os.path.join(save_dir_result, f"binarized_{str(images)}"), thresh)
          cv2.imwrite(os.path.join(save_dir_result, f"measured_{str(images)}"), out)


        if object_name[0] == "circle":
          segmented_cropped_image, opencvImage = k_means_segmentation(cropped_img,im_show=False)
          cv2.imwrite(os.path.join(save_dir_result, f"segmented_cropped_image_{str(images)}"), segmented_cropped_image)
          closing_img = apply_morphology(segmented_cropped_image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)),im_show=False)
          cv2.imwrite(os.path.join(save_dir_result, f"morphology_{str(images)}"), closing_img)
          out, thresh = show_dimentionality_circle(opencvImage, closing_img, im_show=False)
          cv2.imwrite(os.path.join(save_dir_result, f"binarized_{str(images)}"), thresh)
          cv2.imwrite(os.path.join(save_dir_result, f"measured_{str(images)}"), out)
      
    if args["save_tf_detection"]:
        for i in range(len(test_images_np)):
          input_tensor = tf.convert_to_tensor(test_images_np[i], dtype=tf.float32)
          detections, predictions_dict, shapes = detect_fn(input_tensor)


          plot_detections(
              test_images_np[i][0],
              detections['detection_boxes'][0].numpy(),
              detections['detection_classes'][0].numpy().astype(np.uint32)
              + label_id_offset,
              detections['detection_scores'][0].numpy(),
              category_index, figsize=(15, 20), image_name="gif_frame_" + ('%02d' % i) + ".jpg")
      