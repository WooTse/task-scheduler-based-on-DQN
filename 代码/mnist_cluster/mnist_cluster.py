"""A simple MNIST classifier which displays summaries in TensorBoard.
 This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.
It demonstrates the functionality of every TensorBoard dashboard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def train():
  def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
      activations = act(preactivate, name='activation')
      return activations

  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
  print(cluster.as_dict())
  num_workers = len(cluster.as_dict()['worker'])
  num_ps = len(cluster.as_dict()['ps'])
  print('num worker:%d, num ps:%d'%(num_workers,num_ps))
  
  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":
    with tf.device(tf.train.replica_device_setter(cluster=cluster)):
      # Input placeholders
      x = tf.placeholder(tf.float32, [None, 784], name='x-input')
      y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
      image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
      hidden1 = nn_layer(x, 784, 500, 'layer1')
      keep_prob = tf.placeholder(tf.float32)
      dropped = tf.nn.dropout(hidden1, keep_prob)
      y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)
      diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
      cross_entropy = tf.reduce_mean(diff)
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

      global_step = tf.Variable(0,name="global_step",trainable=False)
    with tf.device('/job:worker/task:%d' % FLAGS.task_index):
      # Import data
      mnist = input_data.read_data_sets(FLAGS.data_dir,
                                        one_hot=True,
                                        fake_data=FLAGS.fake_data)
      def feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train or FLAGS.fake_data:
          xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
          k = FLAGS.dropout
        else:
          xs, ys = mnist.test.images, mnist.test.labels
          k = 1.0
        return {x: xs, y_: ys, keep_prob: k}
      opt = tf.train.AdamOptimizer(FLAGS.learning_rate*num_workers)
      opt = tf.train.SyncReplicasOptimizer(
          opt,
          replicas_to_aggregate=num_workers,
          total_num_replicas=num_workers)
      train_step = opt.minimize(cross_entropy, global_step=global_step)
      is_chief = (FLAGS.task_index == 0)
      if is_chief:
        chief_queue_runner=opt.get_chief_queue_runner()
        init_tokens_op=opt.get_init_tokens_op(num_tokens=0)
      init_op = tf.global_variables_initializer()
      init_local = tf.local_variables_initializer()
      sv = tf.train.Supervisor(is_chief=is_chief,
                               init_op=init_op,
                               local_init_op=init_local,
                               summary_op=None,
                               global_step=global_step,
                               saver=None)
      sess_config = tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=False)

      sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)
      if is_chief:
        sv.start_queue_runners(sess,[chief_queue_runner])
        sess.run(init_tokens_op)
      step = 0
      while step < FLAGS.max_steps:
        start_time = time.time()
        _, step = sess.run([train_step, global_step], feed_dict=feed_dict(True))
        duration = time.time() - start_time
        print('global step:%d, using time:%.3f' % (step, duration))
        if step % 50 == 1:  # Record test-set accuracy
          acc = sess.run([accuracy], feed_dict=feed_dict(False))
          print('Accuracy at step %s: %s' % (step, acc))

def main(_):
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=False, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument('--data_dir', type=str, default='./mnist_data',
                      help='Directory for storing input data')
  parser.add_argument('--summaryDir', type=str, default='',
                      help='Summaries log directory')
  parser.add_argument("--ps_hosts", type=str, default="", help="")
  parser.add_argument("--worker_hosts", type=str, default="", help="")
  parser.add_argument("--job_name", type=str, default="", help="One of 'ps', 'worker'")
  # Flags for defining the tf.train.Server
  parser.add_argument("--task_index", type=int, default=0, help="Index of task within the job")
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
