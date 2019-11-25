import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
import tensorflow as tf
import scipy.misc


file = sys.argv[-1]

if file == 'demo.py':
  print ("Error loading video")
  quit

# Define encoder function
def encode(array):
	pil_img = Image.fromarray(array)
	buff = BytesIO()
	pil_img.save(buff, format="PNG")
	return base64.b64encode(buff.getvalue()).decode("utf-8")

video = skvideo.io.vread(file)

answer_key = {}

# Frame numbering starts at 1


with tf.Session() as sess:
	model_saver = tf.train.import_meta_graph('./SavedCarModel/car_model.meta') # Model Path is to be updated here
	model_saver.restore(sess, tf.train.latest_checkpoint('./SavedCarModel'))
	
	graph = tf.get_default_graph()

	image_input = graph.get_tensor_by_name('image_input:0')
	keep_prob = graph.get_tensor_by_name('keep_prob:0')
	logits = graph.get_tensor_by_name('logits:0')

	frame = 1
	for rgb_frame in video:
		# image = scipy.misc.imread('16.png')
		image = rgb_frame
		im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_input: [image]})

		image_shape = (600,800)
	
		im_softmax_car = im_softmax[0][:, 0].reshape(image_shape[0], image_shape[1])
		binary_segmentation_car = np.uint8((im_softmax_car > 0.5).reshape(image_shape[0], image_shape[1]))


		im_softmax_road = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
		binary_segmentation_road = np.uint8((im_softmax_road > 0.5).reshape(image_shape[0], image_shape[1]))

		

		answer_key[frame] = [encode(binary_segmentation_car), encode(binary_segmentation_road)]

		frame += 1

print (json.dumps(answer_key))
