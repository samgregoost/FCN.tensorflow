import os
import shutil

import numpy as np
import cv2
import tensorflow as tf
import tf_utils as utils


absolute_path = os.path.dirname(os.path.realpath(__file__))

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True / False")
tf.flags.DEFINE_string("model_dir", os.path.join(absolute_path, "models"),
                       "Path to VGG model .mat file")
tf.flags.DEFINE_string("model_url",
                       "http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat",
                       "URL to VGG model .mat file")

NUM_OF_CLASSESS = 3
IMAGE_SIZE = 224
labelDict = {"0":"back_ground", "1":"hair", "2":"skin"}

def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("Setting up VGG initialized conv layers...")
    model_data = utils.get_model_data(FLAGS.model_dir, FLAGS.model_url)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]

        pool5 = utils.max_pool_2x2(conv_final_layer)

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3


def predict(img):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")

    pred_annotation, logits = inference(image, keep_probability)

    seg_dress_file = img
    seg_dress_path = seg_dress_file.rsplit(".", 1)[0]

    segmented_dir = seg_dress_path + "_segmented/"
    if os.path.exists(segmented_dir):
        shutil.rmtree(segmented_dir)
    os.makedirs(segmented_dir)

    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored.")
        
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    valid_images = cv2.resize(img,(224,224))
    valid_images = np.expand_dims(valid_images, axis=0)
    
    pred = sess.run(pred_annotation, feed_dict={image: valid_images,
                                                keep_probability: 1.0})
    # workaround for variable scope error
    # ValueError: Variable inference/conv1_1_w already exists, disallowed. Did you mean to set reuse=True in VarScope
    tf.reset_default_graph()

    pred = np.squeeze(pred, axis=3)
    utils.save_image(pred[0].astype(np.uint8), segmented_dir, name="predicted")
    img = cv2.imread(os.path.join(segmented_dir, "predicted.png"), 0)

    k = np.unique(img)
   
    scaledImage = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
   
    backtorgb = cv2.applyColorMap(scaledImage, cv2.COLORMAP_JET)
   
    cv2.imwrite(segmented_dir + 'color.png', backtorgb)
    
#iterate for each label in the predicted image
    for x in np.nditer(k):
	#if label is not background excecute
        if x != 0:
	    #create a mask for the predicted label
            tarImg = img.copy()
            tarImg[img == x] = 255
            tarImg[img != x] = 0
            invertedMask = (255-tarImg)
	    #Apply the mask to the input image
            res = cv2.bitwise_and(valid_images[0],valid_images[0],mask = tarImg)
            res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            labRes = cv2.cvtColor(res, cv2.COLOR_RGB2LAB)

	    #Get the pixel values from the area of the label
            pixelArray = np.empty((0,3),int)
            mheight, mwidth = img.shape[:2]
            for i in range(mwidth):
                for j in range(mheight):
                    if(tarImg[j, i] > 0):
                        pixel = labRes[j, i]
                        pixelArray = np.append(pixelArray, [pixel], axis=0)

	    #Get the masked image in white background
            resCopy = res.copy()
            whiteRes = cv2.bitwise_not(resCopy,resCopy,mask = invertedMask)

	    #save all the images
            try:
                np.savetxt(segmented_dir + labelDict[str(x)] + '_mask.txt', pixelArray, fmt='%d')
                cv2.imwrite(segmented_dir + labelDict[str(x)] + '.png',res)
                cv2.imwrite(segmented_dir + labelDict[str(x)] + '_white.png', whiteRes)
            except:
                print("no key")


