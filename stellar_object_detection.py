"""import tensorflow as tf
from tensorflow import keras
import cv2
import glob
import os
from ultralytics import YOLO
import keras_cv

#DATASET
#load the images with the matching label files, parse boxes and class IDs in for the model to expect
#load in your train images, then tf.data.Dataset.fromtensorslices it
#create a function that loads in the images, reads the boxes and labels from the files, and returns the image and its corresponding box and labels in a dict
#tell tensorflow that the boxes and train images are ragged
#resize and normalize the images inside the above function in ln3

#MODEL
#create your CNN model
#compile without loss
#fit it, then evaluate

train_img_paths = "c:/Users/CHIDIMA/ezinne/yolov11_balanced/train/images"
test_img_paths = "c:/Users/CHIDIMA/ezinne/yolov11_balanced/test/images"
valid_img_paths = "c:/Users/CHIDIMA/ezinne/yolov11_balanced/valid/images"


#to load and process each img, label and box
def load_files(img_path):
    img_height = 512
    img_width = 512

    #tf has to read the file and decode it
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    #resize and normalize the images
    img = tf.image.resize(img, (img_height, img_width))
    img = tf.cast(img, tf.float32) / 255.0

    #get labels from img path and read 'em
    label_path = tf.strings.regex_replace(img_path, "images", "labels")
    label_path = tf.strings.regex_replace(label_path, "\\.jpg", ".txt")

    txt = tf.io.read_file(label_path)
    lines = tf.strings.strip(txt)
    lines = tf.strings.split(lines, "\n")

    # Remove empty lines
    lines = tf.boolean_mask(lines, tf.strings.length(lines) > 0)

    # Split each line into parts (still ragged)
    parts = tf.strings.split(lines, " ")  # RaggedTensor [num_lines, None]

    # Keep only lines with exactly 5 elements
    mask = tf.equal(parts.row_lengths(), 5)
    parts = tf.ragged.boolean_mask(parts, mask)

    # Convert to dense now that all rows have 5 elements
    parts = parts.to_tensor()
    parts = tf.strings.to_number(parts, tf.float32)

    #classes = tf.cast(parts[:, 0], tf.int32)
    classes = tf.cast(parts[:, 0], tf.int32)

    classes = tf.cast(classes, tf.int32)
    tf.rank(classes) == 1
    classes = tf.reshape(classes, [-1])

    cx = parts[:, 1]
    cy = parts[:, 2]
    w  = parts[:, 3]
    h  = parts[:, 4]

    #un-normalizing the coordinates
    cx = cx * img_width
    cy = cy * img_height
    w = w * img_width
    h = h * img_height

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    x1 = tf.reshape(x1, [-1])
    y1 = tf.reshape(y1, [-1])
    x2 = tf.reshape(x2, [-1])
    y2 = tf.reshape(y2, [-1])

    x = x1
    y = y1

    boxes = tf.stack([x1, y1, x2, y2], axis=-1)
    #tf.print(tf.shape(boxes))

    return img, {
        "boxes": tf.cast(boxes, tf.float32),
        "classes": tf.cast(classes, tf.int32),
    }
    #return img, {"boxes": boxes, "classes": classes}
"""



"""MAX_OBJECTS = 20
img_height = 512
img_width = 512


def load_files(img_path):

    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (img_height, img_width))
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.ensure_shape(img, [img_height, img_width, 3])

    label_path = tf.strings.regex_replace(img_path, "images", "labels")
    label_path = tf.strings.regex_replace(label_path, "\\.jpg", ".txt")

    txt = tf.io.read_file(label_path)
    lines = tf.strings.strip(txt)
    lines = tf.strings.split(lines, "\n")
    lines = tf.boolean_mask(lines, tf.strings.length(lines) > 0)

    parts = tf.strings.split(lines, " ")
    mask = tf.equal(parts.row_lengths(), 5)
    parts = tf.ragged.boolean_mask(parts, mask)
    parts = parts.to_tensor()
    parts = tf.strings.to_number(parts, tf.float32)

    classes = tf.cast(parts[:, 0], tf.int32)

    cx = parts[:, 1] * img_width
    cy = parts[:, 2] * img_height
    w  = parts[:, 3] * img_width
    h  = parts[:, 4] * img_height

    boxes = tf.stack([cx, cy, w, h], axis=-1)  # xywh format

    return img, {
        "boxes":   tf.cast(boxes,   tf.float32),
        "classes": tf.cast(classes, tf.int32),
    }



def dataset(img_paths, batch_size, shuffle=False):
    list_of_paths = [
        os.path.join(img_paths, f) 
        for f in os.listdir(img_paths) 
    ]

    ds = tf.data.Dataset.from_tensor_slices(list_of_paths)

    if shuffle:
        ds = ds.shuffle(512)

    ds = ds.map(load_files, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.ragged_batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


train_ds = dataset(train_img_paths, 64, shuffle=True)
test_ds = dataset(test_img_paths, 64)
val_ds = dataset(valid_img_paths, 16)

labels = ['comet', 'galaxy', 'globular_cluster', 'nebula']
num_classes = 4


#model creation
backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_s_backbone_coco")
model = keras_cv.models.YOLOV8Detector(num_classes=num_classes, bounding_box_format="xywh", backbone=backbone)

#model.build((None, 512, 512, 3))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), box_loss='ciou', classification_loss='binary_crossentropy')
model.fit(train_ds, validation_data=val_ds, epochs=30)

print(model.evaluate(test_ds))"""


import tensorflow as tf
import os
import keras_cv

MAX_OBJECTS = 20
IMG_HEIGHT = 512
IMG_WIDTH = 512
NUM_CLASSES = 4
BATCH_SIZE = 16

train_img_paths = r"c:/Users/CHIDIMA/ezinne/yolov11_balanced/train/images"
test_img_paths = r"c:/Users/CHIDIMA/ezinne/yolov11_balanced/test/images"
valid_img_paths = r"c:/Users/CHIDIMA/ezinne/yolov11_balanced/valid/images"


def parse_label_file(label_path):
    txt = tf.io.read_file(label_path)
    txt = tf.strings.strip(txt)

    def no_boxes():
        return tf.zeros((0, 5), dtype=tf.float32)

    def has_boxes():
        lines = tf.strings.split(txt, "\n")
        lines = tf.boolean_mask(lines, tf.strings.length(lines) > 0)

        # Split on whitespace robustly
        parts = tf.strings.split(lines)
        mask = tf.equal(parts.row_lengths(), 5)
        parts = tf.ragged.boolean_mask(parts, mask)

        # Force shape [N, 5]
        parts = parts[:, :5]
        parts = parts.to_tensor(default_value="0", shape=[None, 5])
        parts = tf.strings.to_number(parts, tf.float32)
        return parts

    return tf.cond(tf.equal(tf.strings.length(txt), 0), no_boxes, has_boxes)


def load_files(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.ensure_shape(img, [IMG_HEIGHT, IMG_WIDTH, 3])

    label_path = tf.strings.regex_replace(img_path, "images", "labels")
    label_path = tf.strings.regex_replace(label_path, r"\.(jpg|jpeg|png)$", ".txt")

    parts = parse_label_file(label_path)

    classes = parts[:, 0]

    cx = parts[:, 1] * IMG_WIDTH
    cy = parts[:, 2] * IMG_HEIGHT
    w = parts[:, 3] * IMG_WIDTH
    h = parts[:, 4] * IMG_HEIGHT

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    boxes = tf.stack([x1, y1, x2, y2], axis=-1)

    x1 = tf.clip_by_value(boxes[:, 0], 0, IMG_WIDTH)
    y1 = tf.clip_by_value(boxes[:, 1], 0, IMG_HEIGHT)
    x2 = tf.clip_by_value(boxes[:, 2], 0, IMG_WIDTH)
    y2 = tf.clip_by_value(boxes[:, 3], 0, IMG_HEIGHT)
    boxes = tf.stack([x1, y1, x2, y2], axis=-1)

    boxes = boxes[:MAX_OBJECTS]
    classes = classes[:MAX_OBJECTS]

    num_boxes = tf.shape(boxes)[0]
    pad_size = MAX_OBJECTS - num_boxes

    boxes = tf.pad(boxes, [[0, pad_size], [0, 0]])
    classes = tf.pad(classes, [[0, pad_size]], constant_values=-1)

    boxes = tf.ensure_shape(boxes, [MAX_OBJECTS, 4])
    classes = tf.ensure_shape(classes, [MAX_OBJECTS])

    bounding_boxes = {
        "boxes": tf.cast(boxes, tf.float32),
        "classes": tf.cast(classes, tf.float32),
    }
    
    print("Load mages mg")
    print(img)
    return img, bounding_boxes


def dataset(img_dir, batch_size, shuffle=False):
    list_of_paths = [
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]#brngs out all paths n folder

    ds = tf.data.Dataset.from_tensor_slices(list_of_paths)#produces scalar lst cause argument s 1D tensor

    if shuffle:
        ds = ds.shuffle(len(list_of_paths))

    ds = ds.map(load_files, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = dataset(train_img_paths, BATCH_SIZE, shuffle=True)#true so the model won't learn ds patterns instead
val_ds = dataset(valid_img_paths, BATCH_SIZE)
test_ds = dataset(test_img_paths, BATCH_SIZE)


for images, y in train_ds.take(1):
    print("images:", images.shape)
    print("boxes:", y["boxes"].shape)
    print("classes:", y["classes"].shape)
    print("sample boxes:", y["boxes"][0].numpy())
    print("sample classes:", y["classes"][0].numpy())
    
                                                
backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_s_backbone_coco")

model = keras_cv.models.YOLOV8Detector(
    num_classes=NUM_CLASSES,
    bounding_box_format="xyxy",
    backbone=backbone,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    box_loss="ciou",
    classification_loss="binary_crossentropy",
)

model.fit(train_ds, validation_data=val_ds, epochs=30)
print(model.evaluate(test_ds))