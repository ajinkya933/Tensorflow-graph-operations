import tensorflow as tf

saver = tf.train.import_meta_graph('./VGGnet_fast_rcnn_iter_50000.meta', clear_devices=True)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session()
saver.restore(sess, "./VGGnet_fast_rcnn_iter_50000")

for n in tf.get_default_graph().as_graph_def().node:
	print(n.name)

writer = tf.summary.FileWriter('./log/', sess.graph)
