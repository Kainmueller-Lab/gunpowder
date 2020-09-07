import logging
import os
import numpy as np

from gunpowder.array import ArrayKey, Array
from gunpowder.ext import tensorflow as tf
from gunpowder.nodes.generic_train import GenericTrain
from gunpowder.tensorflow.local_server import LocalServer

logger = logging.getLogger(__name__)

class Train(GenericTrain):
    '''Tensorflow implementation of :class:`gunpowder.nodes.Train`.

    Args:

        graph (``string``):

            Filename of a tensorflow meta-graph storing the tensorflow graph
            containing an optimizer. A meta-graph file can be created by
            running::

                # create tensorflow graph
                ...

                # store it
                tf.train.export_meta_graph(filename='...')

        optimizer (``string`` or function):

            Either the name of the tensorflow operator performing a training
            iteration, or a function that, given the graph of the meta-graph
            file, adds a custom loss and optimizer.

            If a function is given, it should return a tuple ``(loss,
            optimizer)`` of a tensor and an operator representing the loss and
            the optimizer, respectively. In this case, parameter ``loss``
            should be ``None``.

            Example::

                def add_custom_optimizer(graph):

                    # get the output of your graph
                    output = graph.get_tensor_by_name('...')

                    # create your custom loss
                    loss = custom_loss(output)

                    # add an optimizer of your choice
                    optimizer = tf.train.AdamOptimizer().minimize(loss)

                    return (loss, optimizer)

        loss (``string`` or ``None``):

            The name of the tensorflow tensor containing the loss, or ``None``
            if ``optimizer`` is a function.

        inputs (``dict``, ``string`` -> :class:`ArrayKey`):

            Dictionary from the names of input tensors in the network to
            array keys.

        outputs (``dict``, ``string`` -> :class:`ArrayKey`):

            Dictionary from the names of output tensors in the network to array
            keys. New arrays will be generated by this node for each entry (if
            requested downstream).

        gradients (``dict``, ``string`` -> :class:`ArrayKey`):

            Dictionary from the names of output tensors in the network to
            array keys. New arrays containing the gradient of an output with
            respect to the loss will be generated by this node for each entry
            (if requested downstream).

        is_training: (``string``, optional):

            Name of is_training placeholder tensor used to switch between
            training and evaluation mode for dropout/batch norm etc

        summary (``string``, optional):

            The name of the tensorflow tensor containing the tensorboard
            summaries.

        array_specs (``dict``, :class:`ArrayKey` -> :class:`ArraySpec`, optional):

            Used to set the specs of generated arrays (``outputs``). This is
            useful to set the ``voxel_size``, for example, if they differ from
            the voxel size of the input arrays. Only fields that are not
            ``None`` in the given :class:`ArraySpec` will be used.

        save_every (``int``, optional):

            After how many iterations to create a checkpoint to store the
            learnt weights.

        log_dir (``string``, optional):

            Directory for saving tensorboard summaries.

        log_every (``int``, optional):

            After how many iterations to write out tensorboard summaries.

        snapshot_every (``int``, optional):

            Only used in combination with use_tf_data, should have same
            value as in potentially following snapshot node. Pass on inputs
            downstream only in the matching iterations.

    '''

    def __init__(
            self,
            graph,
            optimizer,
            loss,
            inputs,
            outputs,
            gradients,
            is_training=None,
            summary=None,
            array_specs=None,
            save_every=2000,
            use_tf_data=False,
            log_dir='./',
            log_every=1,
            snapshot_every=1):

        super(Train, self).__init__(
            inputs,
            outputs,
            gradients,
            array_specs,
            spawn_subprocess=False)
        self.meta_graph_filename = graph
        self.optimizer_func = None
        self.optimizer_loss_names = None
        self.optimizer = None
        self.loss = None
        self.is_training = is_training
        self.summary = summary
        self.session = None
        self.tf_gradient = {}
        self.graph = None
        self.basic_saver = None
        self.full_saver = None
        self.save_every = save_every
        self.iteration = None
        self.iteration_increment = None
        self.summary_saver = None
        self.log_dir = log_dir
        self.log_every = log_every
        self.snapshot_every = snapshot_every
        self.use_tf_data = use_tf_data
        # Check if optimizer is a str in python 2/3 compatible way.
        if isinstance(optimizer, ("".__class__, u"".__class__)):
            self.optimizer_loss_names = (optimizer, loss)
        else:
            self.optimizer_func = optimizer

        # at least for some versions of tensorflow, the checkpoint name has to
        # start with a . if it is a relative path
        if not os.path.isabs(self.meta_graph_filename):
            self.meta_graph_filename = os.path.join('.', self.meta_graph_filename)

    def start(self):

        # target = LocalServer.get_target()
        # logger.info("Initializing tf session, connecting to %s...", target)

        checkpoint = self.__read_meta_graph()

        if self.summary is not None:
            self.summary_saver = tf.summary.FileWriter(
                self.log_dir, tf.get_default_graph())

        if self.optimizer_func is None:

            self.loss = tf.get_default_graph().get_tensor_by_name(
                self.optimizer_loss_names[1])

        # add symbolic gradients
        for tensor_name in self.gradients:
            tensor = tf.get_default_graph().get_tensor_by_name(
                tensor_name)
            self.tf_gradient[tensor_name] = tf.gradients(
                self.loss,
                [tensor])[0]

        if self.is_training is not None:
            self.is_training = tf.get_default_graph().get_tensor_by_name(
                self.is_training)

        if self.optimizer_func is None:
            # get actual operations/tensors from names
            self.optimizer = tf.get_default_graph().get_operation_by_name(
                self.optimizer_loss_names[0])

        if self.session is None:
            self.session = tf.Session(
                # target=target
            )
            self.__restore_or_init_graph(checkpoint)

        try:
            self.lr = tf.get_default_graph().get_tensor_by_name("learning-rate:0")
        except:
            try:
                self.lr = tf.get_default_graph().get_tensor_by_name("learning-rate/Merge:0")
            except:
                self.lr = None

        self.initialized = True

    def train_step(self, batch, request):

        # initialize tf graph before first step,
        # but after tf.data.Dataset has been created
        if not self.initialized:
            logger.info("first step, loading tf graph...")
            self.tf_data = batch.tf_data
            self.start()

        array_outputs = self.__collect_requested_outputs(request)
        inputs = self.__collect_provided_inputs(batch)
        if self.is_training is not None:
            inputs[self.is_training] = True

        to_compute = {
            'optimizer': self.optimizer,
            'loss': self.loss,
            'iteration': self.iteration_increment}
        if self.lr is not None:
            to_compute['lr'] = self.lr
        to_compute.update(array_outputs)

        # pass on inputs to next gp node that are requested from downstream
        # if snapshot is to be recorded (incurs overhead)
        if self.use_tf_data and \
           self.current_step % self.snapshot_every == 0:
            for _, input_key in self.inputs.items():
                if input_key in request:
                    assert isinstance(input_key, ArrayKey), (
                        "values in inputs dict have to be ArrayKeys (%s)" %
                        input_key)
                    to_compute[input_key] = self.tf_data[str(input_key)]
        # compute outputs, gradients, and update variables
        if self.use_tf_data:
            feed_dict = None
        else:
            feed_dict = inputs
        if self.summary is not None:
            outputs, summaries = self.session.run([to_compute, self.summary], feed_dict=feed_dict)
        else:
            outputs = self.session.run(to_compute, feed_dict=feed_dict)

        for array_key in array_outputs:
            spec = self.spec[array_key].copy()
            spec.roi = request[array_key].roi
            batch.arrays[array_key] = Array(
                outputs[array_key],
                spec)

        if self.use_tf_data and \
           self.current_step % self.snapshot_every == 0:
            for _, input_key in self.inputs.items():
                if input_key in request:
                    spec = self.spec[input_key].copy()
                    spec.roi = request[input_key].roi
                    batch.arrays[input_key] = Array(
                        outputs[input_key],
                        spec)

        batch.loss = outputs['loss']
        batch.iteration = outputs['iteration'][0]
        if self.lr is not None:
            self.current_lr = outputs['lr']
        self.current_step = batch.iteration
        if self.summary is not None and (batch.iteration % self.log_every == 0 or batch.iteration == 1):
            self.summary_saver.add_summary(summaries, batch.iteration)

        if batch.iteration % self.save_every == 0:

            checkpoint_name = (
                self.meta_graph_filename +
                '_checkpoint_%i'%batch.iteration)

            logger.info(
                "Creating checkpoint %s",
                checkpoint_name)

            self.full_saver.save(
                self.session,
                checkpoint_name)

    def stop(self):

        if self.session is not None:

            self.optimizer = None
            self.loss = None
            if self.summary is not None:
                self.summary_saver.close()
            self.session.close()
            self.graph = None
            self.session = None

    def __read_meta_graph(self):

        logger.info("Reading meta-graph...")

        # read the original meta-graph
        if self.use_tf_data:
            input_map = {}
            for k, v in self.inputs.items():
                input_map[k] = self.tf_data[str(v)]
        else:
            input_map = None

        if self.is_training is not None:
            input_map[self.is_training] = True
        tf.train.import_meta_graph(
            self.meta_graph_filename + '.meta',
            input_map=input_map,
            clear_devices=True)

        # add custom gunpowder variables
        with tf.variable_scope('gunpowder'):
            self.iteration = tf.get_variable(
                'iteration',
                shape=1,
                initializer=tf.zeros_initializer,
                trainable=False)
            self.iteration_increment = tf.assign(
                self.iteration,
                self.iteration + 1)

        # Until now, only variables have been added to the graph that are part
        # of every checkpoint. We create a 'basic_saver' for only those
        # variables.
        self.basic_saver = tf.train.Saver(max_to_keep=None)

        # Add custom optimizer and loss, if requested. This potentially adds
        # more variables, not covered by the basic_saver.
        if self.optimizer_func is not None:
            loss, optimizer = self.optimizer_func(self.graph)
            self.loss = loss
            self.optimizer = optimizer

        # We create a 'full_saver' including those variables.
        self.full_saver = tf.train.Saver(max_to_keep=None)

        # find most recent checkpoint
        checkpoint_dir = os.path.dirname(self.meta_graph_filename)
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

        return checkpoint

    def __restore_or_init_graph(self, checkpoint):

        if checkpoint:

            try:
                # Try to restore the graph, including the custom optimizer
                # state (if a custom optimizer was used).
                self.__restore_graph(checkpoint, restore_full=True)

            except tf.errors.NotFoundError:

                # If that failed, we just transitioned from an earlier training
                # without the custom optimizer. In this case, restore only the
                # variables of the original meta-graph and 'gunpowder'
                # variables. Custom optimizer variables will be default
                # initialized.
                logger.info("Checkpoint did not contain custom optimizer "
                            "variables")
                self.__restore_graph(checkpoint, restore_full=False)
        else:

            logger.info("No checkpoint found")

            # initialize all variables
            self.session.run(tf.global_variables_initializer())

        self.current_step = self.session.run(self.iteration)

    def __restore_graph(self, checkpoint, restore_full):

        logger.info("Restoring model from %s", checkpoint)

        if restore_full:

            logger.info("...using a saver for all variables")
            self.full_saver.restore(self.session, checkpoint)

        else:

            # initialize all variables, such that non-basic variables are
            # initialized
            self.session.run(tf.global_variables_initializer())

            logger.info("...using a saver for basic variables only")
            self.basic_saver.restore(self.session, checkpoint)

    def __collect_requested_outputs(self, request):

        array_outputs = {}

        for output_name, array_key in self.outputs.items():
            if array_key in request:
                array_outputs[array_key] = output_name

        for output_name, array_key in self.gradients.items():
            if array_key in request:
                array_outputs[array_key] = self.tf_gradient[output_name]

        return array_outputs

    def __collect_provided_inputs(self, batch):

        inputs = {}

        for input_name, input_key in self.inputs.items():
            if isinstance(input_key, ArrayKey):
                if input_key in batch.arrays:
                    inputs[input_name] = batch.arrays[input_key].data
                else:
                    logger.warn("batch does not contain %s, input %s will not "
                                "be set", input_key, input_name)
            elif isinstance(input_key, np.ndarray):
                inputs[input_name] = input_key
            elif isinstance(input_key, str):
                inputs[input_name] = getattr(batch, input_key)
            else:
                raise Exception(
                    "Unknown network input key {}, can't be given to "
                    "network".format(input_key))

        return inputs
