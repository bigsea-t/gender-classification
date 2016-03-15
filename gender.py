import time

import numpy as np
import tensorflow as tf
import tensorflow.models.rnn
import reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", None, "data_path")
flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")

FLAGS = flags.FLAGS


class GenderModel(object):
    """Gender Classification model."""

    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size
        num_classes = config.n_classes

        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.float32, [batch_size, num_classes])

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                    lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size])
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        inputs = [tf.squeeze(input_, [1])
                  for input_ in tf.split(1, num_steps, inputs)]
        outputs, state = tf.models.rnn.rnn.rnn(cell, inputs, initial_state=self._initial_state)

        pooled = tf.reduce_max(tf.pack(outputs), 0, name='pooling')

        softmax_w = tf.get_variable("softmax_w", [size, num_classes])
        softmax_b = tf.get_variable("softmax_b", [num_classes])
        # y = tf.nn.softmax(tf.matmul(pooled, softmax_w) + softmax_b, "y")
        logits = tf.matmul(pooled, softmax_w) + softmax_b
        loss = tf.nn.softmax_cross_entropy_with_logits(logits, self._targets)
        self._cost = cost = tf.reduce_mean(loss)

        # self._cost = cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logit, self._targets))
        tf.scalar_summary("xentropy", cost)

        # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self._targets, 1))
        # self._accuracy = accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # accuracy_summary = tf.scalar_summary("accuracy", accuracy)

        self._summary = tf.merge_all_summaries()

        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))


    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    # @property
    # def accuracy(self):
    #     return self._accuracy

    @property
    def summary(self):
        return self._summary


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 1
    num_steps = 40 #200 # should be same as max_len of words
    hidden_size = 200
    max_epoch = 1 #4
    max_max_epoch = 2 #13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    # vocab_size = 10000
    vocab_size = 89972
    n_classes = 2


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 1
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


def run_epoch(session, m, data, eval_op, verbose=False):
    """Runs the model on the given data."""
    posts, labels = data
    n_docs, n_words = posts.shape
    epoch_size = n_docs // m.batch_size
    costs = 0.0
    accs = 0.0
    iters = 0
    state = m.initial_state.eval()
    for step, (texts, label) in enumerate(reader.data_iterator(data, m.batch_size,
                                                      m.num_steps)):
        print('texts', texts.shape)
        print('label', label.shape)
        cost, state, _, summary = session.run([m.cost, m.final_state, eval_op, m.summary],
                                                                 {m.input_data: texts,
                                                                     m.targets: label,
                                                                     m.initial_state: state})
        acc = 0
        costs += cost
        accs += acc
        iters += m.batch_size

        if iters > 3:
            return costs / iters, accs / iters, summary
        if verbose and step % (epoch_size // 10) == 0:
            print("%.3f xentropy: %.3f " %
                  (step * 1.0 / epoch_size, costs / iters))

    return costs / iters, accs / iters, summary


def get_config():
    if FLAGS.model == "small":
        return SmallConfig()
    elif FLAGS.model == "medium":
        return MediumConfig()
    elif FLAGS.model == "large":
        return LargeConfig()
    elif FLAGS.model == "test":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to data directory")

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    # eval_config.num_steps = 1

    raw_data = reader.converted_data(FLAGS.data_path, max_len=config.num_steps, min_nwords=200)

    train_data, valid_data, test_data = reader.split_rawdata(raw_data)


    sess = tf.InteractiveSession()
    writer = tf.train.SummaryWriter("/tmp/mnist_logs", sess.graph_def)

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = GenderModel(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = GenderModel(is_training=False, config=config)
            mtest = GenderModel(is_training=False, config=eval_config)

        tf.initialize_all_variables().run()

        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_error, train_acc, summary = run_epoch(session, m, train_data, m.train_op, verbose=True)
            writer.add_summary(summary, i)
            print("Epoch: %d Train Error: %.3f" % (i + 1, train_error))

            valid_error, valid_acc, summary = run_epoch(session, mvalid, valid_data, tf.no_op())
            writer.add_summary(summary, i)
            print("Epoch: %d Valid Error: %.3f" % (i + 1, valid_error))

        test_err, test_acc, summary = run_epoch(session, mtest, test_data, tf.no_op())
        print("Test Accuracy %.3f", test_acc)


if __name__ == "__main__":
    tf.app.run()