import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.Session()

with tf.gfile.GFile("../../../data/images/lena.jpg", "rb") as file:
    image_raw = file.read()
    image_tensor = tf.image.decode_jpeg(image_raw)
    
    batch = tf.expand_dims(
        tf.image.convert_image_dtype(image_tensor, tf.float32), 0)
    
    boxes = tf.constant([[[0.1, 0.2, 0.75, 0.7], 
                          [0.5, 0.45, 0.55, 0.70]]]) # [ymin xmin ymax xmax] 
    print("boxes.shape:", boxes.shape)               # dims: (batch, number, 4)
    with_box = tf.image.draw_bounding_boxes(batch, boxes)
    batch_result = sess.run(with_box)

    plt.figure("With bounding box")
    plt.imshow(batch_result[0])
    
    # slice random box
    begin, size, box_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(image_tensor), bounding_boxes=boxes)
    
    sliced_region = tf.image.draw_bounding_boxes(batch, box_for_draw)
    sliced_image = tf.slice(image_tensor, begin, size)
    s_region, s_image = sess.run([sliced_region, sliced_image])
    
    plt.figure("Sliced region")
    plt.imshow(s_region[0])
    
    plt.figure("Sliced image")
    plt.imshow(s_image)
    
plt.show()
sess.close()
