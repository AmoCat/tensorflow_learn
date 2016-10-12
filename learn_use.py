import tensorflow as tf
import numpy

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.add(input1,input2)

with tf.Session() as sess:
    print sess.run([output],feed_dict = {input1:[3.],input2:[3.]}) 
