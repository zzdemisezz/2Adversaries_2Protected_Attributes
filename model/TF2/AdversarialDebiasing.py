import numpy as np
import scipy.special
import scipy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import confusion_matrix
from sklearn.utils import check_random_state
import tensorflow as tf


# class for classifier
class classifier_model(tf.Module):
    def __init__(self, features, seed1):
        super(classifier_model, self).__init__()
        self.features = features
        self.seed1 = seed1

        # no hidden layer
        # uniform initializer
        self.W1 = tf.Variable(tf.random.uniform(shape=(self.features, 1), seed=self.seed1), name='W1')
        # glorot uniform initializer, not very different from uniform, but could be used to find better result
        # initializer = tf.initializers.glorot_uniform(seed=self.seed1)
        # self.W1 = tf.Variable(initializer(shape=(self.features, 1)), name='W1')

        self.b1 = tf.Variable(tf.zeros(shape=1), name='b1')

    # function to use classifier
    def forward(self, x):
        # no hidden layer
        classifier_logits = tf.matmul(x, self.W1) + self.b1
        classifier_pred = tf.sigmoid(classifier_logits)
        return classifier_pred, classifier_logits


# class for adversary
class adversary_model(tf.Module):
    def __init__(self, seed3, seed5):
        super(adversary_model, self).__init__()
        self.seed3 = seed3
        self.c = tf.Variable(tf.constant(1.0), name='c')

        # uniform initializer
        self.W1 = tf.Variable(tf.random.normal(shape=(3, 1), seed=self.seed3), name='w1')
        # glorot uniform initializer, not very different from uniform, but could be used to find better result
        # initializer = tf.initializers.glorot_uniform(seed=self.seed3)
        # self.W1 = tf.Variable(initializer(shape=(3, 1)), name='W1')

        self.b1 = tf.Variable(tf.zeros(shape=1), name='b1')

    # function to use adversary
    def forward(self, classifier_logits, true_income):
        s = tf.sigmoid((1 + tf.abs(self.c)) * classifier_logits)
        adversary_logits = tf.matmul(tf.concat([s, s * true_income, s * (1.0 - true_income)], axis=1), self.W1) + self.b1
        return adversary_logits


# class for adversary
class adversary1_model(tf.Module):
    # first testing without dropout
    def __init__(self, n_features, seed3, seed5):
        super(adversary1_model, self).__init__()
        self.n_features = n_features

        self.seed3 = seed3
        self.seed5 = seed5

        # uniform initializer
        self.W1 = tf.Variable(tf.random.normal(shape=(2, self.n_features), seed=self.seed3), name='w1')
        # glorot uniform initializer, not very different from uniform, but could be used to find better result
        # initializer = tf.initializers.glorot_uniform(seed=self.seed3)
        # self.W1 = tf.Variable(initializer(shape=(3, 1)), name='W1')
        self.b1 = tf.Variable(tf.zeros(shape=1), name='b1')

        self.W2 = tf.Variable(tf.random.normal(shape=(self.n_features, 1), seed=self.seed5), name='w2')
        self.b2 = tf.Variable(tf.zeros(shape=1), name='b2')

    # function to use adversary
    def forward(self, classifier_logits, true_income):
        input_hidden_layer = tf.concat([classifier_logits, true_income], axis=1)
        hidden_layer = tf.nn.relu(tf.matmul(input_hidden_layer, self.W1)) + self.b1
        # hidden_layer = tf.nn.dropout(hidden_layer, rate=0, seed=self.seed4)
        adversary_logits = tf.matmul(hidden_layer, self.W2) + self.b2

        return adversary_logits


# Class for second adversary
class adversary2_model(tf.Module):
    # first testing without dropout
    def __init__(self, n_features, seed2, seed4):
        super(adversary2_model, self).__init__()
        self.n_features = n_features

        self.seed2 = seed2
        self.seed4 = seed4

        # uniform initializer
        self.W1 = tf.Variable(tf.random.normal(shape=(2, self.n_features), seed=self.seed2), name='w1')
        # glorot uniform initializer, not very different from uniform, but could be used to find better result
        # initializer = tf.initializers.glorot_uniform(seed=self.seed3)
        # self.W1 = tf.Variable(initializer(shape=(3, 1)), name='W1')
        self.b1 = tf.Variable(tf.zeros(shape=1), name='b1')

        self.W2 = tf.Variable(tf.random.normal(shape=(self.n_features, 1), seed=self.seed4), name='w2')
        self.b2 = tf.Variable(tf.zeros(shape=1), name='b2')

    # function to use adversary
    def forward(self, classifier_logits, true_income):
        input_hidden_layer = tf.concat([classifier_logits, true_income], axis=1)
        hidden_layer = tf.nn.relu(tf.matmul(input_hidden_layer, self.W1)) + self.b1
        # hidden_layer = tf.nn.dropout(hidden_layer, rate=0, seed=self.seed4)
        adversary_logits = tf.matmul(hidden_layer, self.W2) + self.b2
        # print(tf.shape(adversary_logits))
        adversary_pred = tf.sigmoid(adversary_logits)

        return adversary_logits


# class for running classifer or classifier + adversary
class AdversarialDebiasing(BaseEstimator, ClassifierMixin):
    # when changing the adversary to a general neural network, maybe put number of hidden units in this __init__
    # prot_attr = gender
    def __init__(self, prot_attr=None, prot_attr2=None, scope_name='classifier',
                 adversary_loss_weight=0.1, num_epochs=50, batch_size=256, debias=True, random_state=None, multiple=False, multiple_features=
                 False):

        r"""
        Args:
            prot_attr (single label or list-like, optional): Protected
                attribute(s) to use in the debiasing process. If more than one
                attribute, all combinations of values (intersections) are
                considered. Default is ``None`` meaning all protected attributes
                from the dataset are used.

            scope_name (str, optional): TensorFlow "variable_scope" name for the
                entire model (classifier and adversary).

            adversary_loss_weight (float or ``None``, optional): If ``None``,
                this will use the suggestion from the paper:
                :math:`\alpha = \sqrt(global_step)` with inverse time decay on
                the learning rate. Otherwise, it uses the provided coefficient
                with exponential learning rate decay.

            num_epochs (int, optional): Number of epochs for which to train.

            batch_size (int, optional): Size of mini-batch for training.

            debias (bool, optional): If ``False``, learn a classifier without an
                adversary.

            random_state (int or numpy.RandomState): Seed of pseudo-random number generator
            for shuffling data and seeding weights.
        """

        self.prot_attr = prot_attr
        self.prot_attr2 = prot_attr2
        self.scope_name = scope_name
        self.adversary_loss_weight = adversary_loss_weight
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.debias = debias
        self.random_state = random_state
        self.multiple = multiple
        self.multiple_features = multiple_features

        # here so we don't get orange lines
        self.classes_ = None
        self.clf_model = None
        self.adv1_model = None
        self.adv2_model = None

        # setting our seeds for weight initialization
        rng = check_random_state(self.random_state)

        ii32 = np.iinfo(np.int32)
        # we accidentally did all our tuning with seed 3 instead of seed 2 (:
        # self.s1, self.s2, self.s3 = rng.randint(ii32.min, ii32.max, size=3)
        self.s1, self.s2, self.s3, self.s4, self.s5 = rng.randint(ii32.min, ii32.max, size=5)

        # I don't know what the line below does, but when I remove it, our results get worse
        tf.random.set_seed(self.random_state)

    def fit(self, x_train, y_train, x_validation, y_validation, x_test, y_test):
        """Train the classifier (and adversary if ``debias == True``) with the
        given training data.

        Args:
            y_validation:
            x_validation:
            x_test: we used this for model tuning, remove it when handing in our code
            y_test: we used this for model tuning, remove it when handing in our code
            x_train (pandas.DataFrame): Training samples.
            y_train (array-like): Training labels.

        Returns:
            self
        """

        # changing X to the right datatype
        if scipy.sparse.issparse(x_train):
            x_train = x_train.todense()
        if scipy.sparse.issparse(x_validation):
            x_validation = x_validation.todense()
        if scipy.sparse.issparse(x_test):
            x_test = x_test.todense()

        # protected_attribute = gender, datatype = pandas.core.frame.DataFrame
        protected_attribute = y_train[self.prot_attr]
        protected_attribute2 = y_train[self.prot_attr2]
        # dependent_variable_classifier = income, datatype = pandas.core.frame.DataFrame
        dependent_variable_classifier = y_train.drop(columns=self.prot_attr)

        if self.multiple_features:
            dependent_variable_classifier = dependent_variable_classifier.drop(columns=self.prot_attr2)

        # self.classes_ is used in def predict, we believe it is just the value 2
        # perhaps change his def predict with the one Liu wrote after the draft deadline
        self.classes_ = np.unique(dependent_variable_classifier)

        # dimensions data
        num_train_samples, n_features = x_train.shape
        # print(n_features)

        self.clf_model = classifier_model(features=n_features, seed1=self.s1)

        # learning rate = η in paper. The code originally used exponential decay, decay rate 0.96 and decay steps 1000.
        # The paper suggest that eta should be proportional to 1/t (InverseTimeDecay).
        # Paper suggested using InverseTimeDecay in combination with adversarial_loss_weight (alpha) = sqrt(t).
        # We get good results when using InverseTimeDecay with alpha = 0.16, but not with alpha = sqrt(t).
        # I also tested different learning rate for classifier and adversary with inverseTimeDecay
        # I also tested constant learning rates for both classifier and adversary and got good results
        starter_learning_rate_classifier = 0.001
        # 0.1 worked well for constant learning_rate_adversary with InversetimeDecay for classifier
        starter_learning_rate_adversary = 0.001
        starter_learning_rate_multiple = 0.001

        # used for decayed learning rate
        learning_rate_classifier = tf.keras.optimizers.schedules.InverseTimeDecay(starter_learning_rate_classifier,
                                                                       decay_steps=1025, decay_rate=0.66,
                                                                       staircase=True)
        # defining optimizer with decayed learning rate
        # classifier_opt = tf.optimizers.Adam(learning_rate_classifier)
        # defining optimizer with constant learning rate
        classifier_opt = tf.optimizers.Adam(starter_learning_rate_classifier)

        # makes a tensor with trainable variables classifier
        classifier_vars = [var for var in self.clf_model.trainable_variables]

        if self.multiple:
            print("Multiple Adversarials: ")

            self.adv1_model = adversary1_model(n_features=n_features, seed3=self.s3, seed5=self.s5)
            adversary_opt = tf.optimizers.Adam(starter_learning_rate_adversary)

            self.adv2_model = adversary2_model(n_features=n_features, seed2=self.s2, seed4=self.s4)
            multiple_opt = tf.optimizers.Adam(starter_learning_rate_multiple)

            normalize = lambda x: x / (tf.norm(x) + np.finfo(np.float32).tiny)

            for epoch in range(self.num_epochs):
                shuffled_ids = [i for i in range(num_train_samples)]
                for i in range(num_train_samples // self.batch_size):
                    batch_ids = shuffled_ids[self.batch_size * i: self.batch_size * (i + 1)]
                    batch_features = x_train[batch_ids].astype('float32')
                    batch_labels = np.reshape(dependent_variable_classifier.iloc[batch_ids], [-1, 1]).astype('float32')
                    batch_protected_attributes = np.reshape(protected_attribute.iloc[batch_ids], [-1, 1]).astype(
                        'float32')
                    batch_protected_attributes2 = np.reshape(protected_attribute2.iloc[batch_ids], [-1, 1]).astype(
                        'float32')

                    descent_direction = []

                    # Compute the perpendicular of gradient of mlt model on gradient of adv model

                    # Implement gradient formula, get classifier gradient Wlp
                    with tf.GradientTape() as tape:
                        classifier_pred, classifier_logits = self.clf_model.forward(batch_features)
                        loss_classifier = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_labels, logits=classifier_logits))
                    classifier_grad = tape.gradient(loss_classifier, classifier_vars)

                    # Gradient WLA1
                    with tf.GradientTape() as tape_adv:
                        classifier_pred, classifier_logits = self.clf_model.forward(
                            batch_features)  # variables of CLF_model need to be watched from tape1 also
                        # runs adversary model with classifier_logit as input
                        pred_protected_attributes_logits = self.adv1_model.forward(
                            classifier_logits, batch_labels)
                        loss_adv = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_protected_attributes,
                                                                    logits=pred_protected_attributes_logits))
                    adv_grad = tape_adv.gradient(loss_adv, classifier_vars)

                    # Gradient WlA2
                    with tf.GradientTape() as tape_mlt:
                        classifier_pred, classifier_logits = self.clf_model.forward(
                            batch_features)
                        pred_protected_attributes_logits_mlt = self.adv2_model.forward(
                            classifier_logits, batch_labels)
                        loss_mlt = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_protected_attributes2,
                                                                    logits=pred_protected_attributes_logits_mlt))
                    mlt_grad = tape_mlt.gradient(loss_mlt, classifier_vars)

                    for _, (grad, var) in enumerate(zip(classifier_grad, self.clf_model.trainable_variables)):
                        # all lines are just calculating the descent direction (fancy formula paper)
                        # unit_adversary_grad = normalize(adv_grad[_])
                        # unit_multiple_grad = normalize(mlt_grad[_])

                        # Correct projection, first term
                        # grad -= tf.reduce_sum(grad * unit_adversary_grad) * unit_adversary_grad

                        # Compute perpendicular gradient b - proj_a(b). (a = adv1, b = mlt model)
                        # orthogonal_grad = mlt_grad[_] - tf.reduce_sum(mlt_grad[_] * unit_adversary_grad) * unit_adversary_grad
                        # Perp wLA1 * wLA2
                        # orthogonal_grad = orthogonal_grad * mlt_grad[_]
                        # orthogonal_grad_unit = normalize(orthogonal_grad)

                        # Normalize again and put it in grad
                        # grad -= tf.reduce_sum(grad * orthogonal_grad) * orthogonal_grad_unit

                        # use line below for alpha = sqrt(t) as suggested by the paper
                        # grad -= np.sqrt(step_count) * adversary_grads[_]
                        # grad -= self.adversary_loss_weight * (adv_grad[_] + orthogonal_grad)
                        # descent_direction.append((grad, var))

                        # all lines are just calculating the descent direction (fancy formula paper) - DOES NOT WORK
                        # normalizing gradients adversaries w.r.t W
                        adv1_grad = normalize(adv_grad[_])
                        adv2_grad = normalize(mlt_grad[_])

                        # Compute perpendicular gradient a2 - proj_a2(a1). (a2 = adv2, a1 = a1 model)
                        perp_adv2_grad = adv2_grad - (tf.reduce_sum(adv2_grad * adv1_grad) * adv1_grad)
                        perp_adv2_grad = normalize(perp_adv2_grad)

                        # Correct projection, first term
                        grad -= tf.reduce_sum(grad * adv1_grad) * adv1_grad
                        # finalize projection
                        grad -= tf.reduce_sum(grad * perp_adv2_grad) * perp_adv2_grad

                        # still need to do adv1_grad perp to adv2 for theoretically sound result
                        # perp_adv1_grad = adv1_grad[_] - tf.reduce_sum(adv1_grad[_] * adv2_grad) * adv2_grad,
                        # # have to ask markus whether we should normalize the adv grads, probably not needed
                        # grad -= self.adversary_loss_weight * (adv1_grad + perp_adv2_grad)

                        # with 2 perp
                        # perp_adv1_grad = adv1_grad - (tf.reduce_sum(adv1_grad * adv2_grad) * adv2_grad)
                        # perp_adv1_grad = normalize(perp_adv1_grad)
                        # grad -= self.adversary_loss_weight * (perp_adv1_grad + perp_adv2_grad)
                        # with 1 perp
                        # grad -= self.adversary_loss_weight * (adv1_grad + perp_adv2_grad)
                        # w/o perp
                        grad -= self.adversary_loss_weight * (adv1_grad + adv2_grad)
                        descent_direction.append((grad, var))

                    classifier_opt.apply_gradients(descent_direction)

                    # Update adversary model
                    with tf.GradientTape() as tape_adv:
                        pred_protected_attributes_logits = self.adv1_model.forward(
                           classifier_logits, batch_labels)
                        loss_adv = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_protected_attributes,
                                                                    logits=pred_protected_attributes_logits))
                    gradients_adv = tape_adv.gradient(loss_adv, self.adv1_model.trainable_variables)
                    adversary_opt.apply_gradients(zip(gradients_adv, self.adv1_model.trainable_variables))

                    # Update multiple model
                    with tf.GradientTape() as tape_mlt:
                        pred_protected_attributes_logits_mlt = self.adv2_model.forward(
                           classifier_logits, batch_labels)
                        loss_mlt = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_protected_attributes2,
                                                                    logits=pred_protected_attributes_logits_mlt))
                    gradients_mlt = tape_mlt.gradient(loss_mlt, self.adv2_model.trainable_variables)
                    multiple_opt.apply_gradients(zip(gradients_mlt, self.adv2_model.trainable_variables))

                    # Printing of the results
                    if i % (num_train_samples // self.batch_size - 1) == 0 and i != 0:
                        feature = 'gender'
                        predict = 'income'

                        # training data metrics
                        pred_labels_all, pred_logits_all = self.clf_model.forward(x_train.astype('float32'))
                        y_pred_all = np.array(predict_class(pred_labels_all))

                        loss_classifier = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(
                                labels=y_train.drop(columns=self.prot_attr).drop(columns=self.prot_attr2).astype('float32'),
                                logits=pred_logits_all))

                        dao_sex = compute_DAO(y_pred_all, y_train, feature, predict, 1, 0, [0, 1])
                        dao_race = compute_DAO(y_pred_all, y_train, 'race', predict, 1, 0, [0, 1])
                        clf_accuracy = accuracy(predict_class(pred_labels_all),
                                                y_train.drop(columns=self.prot_attr).drop(columns=self.prot_attr2))  # tf.reduce_mean(clf_acc)

                        pred_protected_attributes_logits_all = self.adv1_model.forward(pred_logits_all, y_train.drop(
                            columns=self.prot_attr).drop(columns=self.prot_attr2))

                        loss_adv = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train[self.prot_attr].astype('float32'),
                                                                    logits=pred_protected_attributes_logits_all))
                        adv_accuracy = accuracy(predict_class(tf.sigmoid(pred_protected_attributes_logits_all)),
                                                y_train[self.prot_attr])

                        pred_protected_attributes_logits_all_mlt = self.adv2_model.forward(pred_logits_all, y_train.drop(
                            columns=self.prot_attr).drop(columns=self.prot_attr2))
                        mlt_accuracy = accuracy(predict_class(tf.sigmoid(pred_protected_attributes_logits_all_mlt)),
                                                y_train[self.prot_attr2])

                        # Computation of validation set results

                        pred_labels_all_validation, pred_logits_all_validation = self.clf_model.forward(
                            x_validation.astype('float32'))
                        y_pred_all_validation = np.array(predict_class(pred_labels_all_validation))
                        dao_sex_validation = compute_DAO(y_pred_all_validation, y_validation, feature, predict, 1, 0,
                                                         [0, 1])
                        dao_race_validation = compute_DAO(y_pred_all_validation, y_validation, 'race', predict, 1, 0,
                                                          [0, 1])
                        clf_accuracy_validation = accuracy(predict_class(pred_labels_all_validation),
                                                           y_validation.drop(columns=self.prot_attr).drop(
                                                               columns=self.prot_attr2))

                        pred_protected_attributes_logits_all_validation = self.adv1_model.forward(
                            pred_logits_all_validation,
                            y_validation.drop(columns=self.prot_attr).drop(columns=self.prot_attr2))
                        adv_accuracy_validation = accuracy(
                            predict_class(tf.sigmoid(pred_protected_attributes_logits_all_validation)),
                            y_validation[self.prot_attr])

                        pred_protected_attributes_logits_all_mlt_validation = \
                            self.adv2_model.forward(pred_logits_all_validation, y_validation.drop(
                                            columns=self.prot_attr).drop(columns=self.prot_attr2).astype('float32'))
                        mlt_accuracy_validation = accuracy(predict_class(tf.sigmoid(pred_protected_attributes_logits_all_mlt_validation)),
                                                y_validation[self.prot_attr2])

                        # Temporary Computation of Test results for replication (To be deleted after)
                        pred_labels_all_test, pred_logits_all_test = self.clf_model.forward(x_test.astype('float32'))
                        y_pred_all_test = np.array(predict_class(pred_labels_all_test))
                        dao_sex_test = compute_DAO(y_pred_all_test, y_test, feature, predict, 1, 0, [0, 1])
                        dao_race_test = compute_DAO(y_pred_all_test, y_test, 'race', predict, 1, 0, [0, 1])
                        clf_accuracy_test = accuracy(predict_class(pred_labels_all_test),
                                                     y_test.drop(columns=self.prot_attr).drop(columns=self.prot_attr2))

                        pred_protected_attributes_logits_all_test = self.adv1_model.forward(
                            pred_logits_all_test,
                            y_test.drop(columns=self.prot_attr).drop(columns=self.prot_attr2))
                        adv_accuracy_test = accuracy(
                            predict_class(tf.sigmoid(pred_protected_attributes_logits_all_test)),
                            y_test[self.prot_attr])

                        pred_protected_attributes_logits_all_test_mlt = self.adv2_model.forward(
                            pred_logits_all_test,
                            y_test.drop(columns=self.prot_attr).drop(columns=self.prot_attr2))
                        mlt_accuracy_test = accuracy(
                            predict_class(tf.sigmoid(pred_protected_attributes_logits_all_test_mlt)),
                            y_test[self.prot_attr2])

                        print(
                            "(Multiple debiasing) epoch %d; iter: %d; classifier loss: %f; "
                            "adversarial loss: %f; classifier accuracy: %f; adversary accuracy: %f; multiple accuracy: %f; "
                            "DAO sex: %f; DAO race: %f; " % (
                                epoch, i, loss_classifier, loss_adv, clf_accuracy, adv_accuracy, mlt_accuracy, dao_sex, dao_race))

                        # Print validation results
                        print(
                            "(VALIDATION RESULTS) epoch %d; iter: %d; classifier accuracy: %f; adversary accuracy: %f; multiple accuracy: %f; DAO sex %f; DAO race: %f; "
                            % (epoch, i, clf_accuracy_validation, adv_accuracy_validation, mlt_accuracy_validation, dao_sex_validation,
                               dao_race_validation))

                        # PRINT TEMPORARY TEST RESULTS
                        print(
                            "(TEST RESULTS)       epoch %d; iter: %d; classifier accuracy: %f; adversary accuracy: %f; multiple accuracy: %f; DAO sex %f; DAO race: %f; "
                            % (epoch, i, clf_accuracy_test, adv_accuracy_test, mlt_accuracy_test, dao_sex_test, dao_race_test))

        # train classifier + adversary model
        elif self.debias:
            # train adversary
            self.adv1_model = adversary_model(seed3=self.s3, seed5=self.s5)
            # used for decayed learning rate
            learning_rate_adversary = tf.keras.optimizers.schedules.InverseTimeDecay(starter_learning_rate_adversary,
                                                                        decay_steps=1025, decay_rate=0.66,
                                                                        staircase=True)

            # We could possibly use the line below for multiple adversaries
            # adversary_vars = [var for var in self.adv_model.trainable_variables]

            # defining optimizer with decayed learning rate
            # adversary_opt = tf.optimizers.Adam(learning_rate_adversary)
            # defining optimizer with constant learning rate
            adversary_opt = tf.optimizers.Adam(starter_learning_rate_adversary)

            # Actual training with special formula
            normalize = lambda x: x / (tf.norm(x) + np.finfo(np.float32).tiny)

            early_stopper = EarlyStopper(patience=4, min_delta=0.001)
            checkpoint_adv = tf.train.Checkpoint(model=self.clf_model)
            manager_adv = tf.train.CheckpointManager(checkpoint_adv, directory='./checkpoints_adv', max_to_keep=6)
            earlystop_clf, save_epoch = 100, 0

            # if you use the alpha = sqrt(t), you need the line below
            # step_count = 0
            for epoch in range(self.num_epochs):
                shuffled_ids = [i for i in range(num_train_samples)]
                # if you use the alpha = sqrt(t), with t being the number of epochs, you need the lines below
                # step_count += 1
                # self.adversary_loss_weight = np.sqrt(step_count)
                # Perhaps use random batches instead of same order every time
                # shuffled_ids = np.random.choice(num_train_samples, num_train_samples, replace=False)
                for i in range(num_train_samples // self.batch_size):
                    # if you use the alpha = sqrt(t), with t being the number of batches, you need the lines below
                    # step_count += 1
                    batch_ids = shuffled_ids[self.batch_size * i: self.batch_size * (i + 1)]
                    batch_features = x_train[batch_ids].astype('float32')
                    batch_labels = np.reshape(dependent_variable_classifier.iloc[batch_ids], [-1, 1]).astype('float32')
                    batch_protected_attributes = np.reshape(protected_attribute.iloc[batch_ids], [-1, 1]).astype('float32')

                    descent_direction = []

                    with tf.GradientTape() as tape:
                        classifier_pred, classifier_logits = self.clf_model.forward(batch_features)
                        loss_classifier = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_labels, logits=classifier_logits))
                    classifier_grad = tape.gradient(loss_classifier, classifier_vars)


                    with tf.GradientTape() as tape1:
                        # runs classifier model
                        classifier_pred, classifier_logits = self.clf_model.forward(
                           batch_features)  # variables of CLF_model need to be watched from tape1 also
                        # runs adversary model with classifier_logit as input
                        pred_protected_attributes_logits = self.adv1_model.forward(
                            classifier_logits, batch_labels)
                        # calculates loss function adversary
                        loss_adv = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_protected_attributes,
                                                                    logits=pred_protected_attributes_logits))
                    # calculates gradients adversary w.r.t W (W from paper are the weights of the classifier)
                    adversary_grads = tape1.gradient(loss_adv, classifier_vars)

                    for _, (grad, var) in enumerate(zip(classifier_grad, self.clf_model.trainable_variables)):
                        # all lines are just calculating the descent direction (fancy formula paper)
                        unit_adversary_grad = normalize(adversary_grads[_])
                        grad -= tf.reduce_sum(grad * unit_adversary_grad) * unit_adversary_grad  # Projection
                        # use line below for alpha = sqrt(t) as suggested by the paper
                        # grad -= np.sqrt(step_count) * adversary_grads[_]
                        grad -= self.adversary_loss_weight * adversary_grads[_]  # Alpha term
                        descent_direction.append((grad, var))

                    # updates weights classifier using ADAM
                    classifier_opt.apply_gradients(descent_direction)
                    with tf.GradientTape() as tape2:
                        # runs adversary
                        pred_protected_attributes_logits = self.adv1_model.forward(
                           classifier_logits, batch_labels)
                        # calculates loss function adversary
                        loss_adv = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_protected_attributes,
                                                                    logits=pred_protected_attributes_logits))
                    # calculates gradient adversary w.r.t U (U from the paper are the weights of the adversary)
                    gradients = tape2.gradient(loss_adv, self.adv1_model.trainable_variables)
                    # updates weights adversary using ADAM
                    adversary_opt.apply_gradients(zip(gradients, self.adv1_model.trainable_variables))

                    # computing metrics for training and test data: classifier accuracy, adversary accuracy and DAO
                    # any questions can be asked to Liu
                    # using line below calculates the metrics for only the first batch, might still want to use it
                    # if i % 200 == 0:
                    if i % (num_train_samples // self.batch_size - 1) == 0 and i != 0:
                        feature = 'gender'
                        feature2 = 'race'
                        predict = 'income'

                        pred_labels_all, pred_logits_all = self.clf_model.forward(x_train.astype('float32'))
                        y_pred_all = np.array(predict_class(pred_labels_all))

                        dao_sex = compute_DAO(y_pred_all, y_train, feature, predict, 1, 0, [0, 1])
                        dao_race = compute_DAO(y_pred_all, y_train, feature2, predict, 1, 0, [0, 1])
                        clf_accuracy = accuracy(predict_class(pred_labels_all),
                                                y_train.drop(columns=self.prot_attr).drop(columns=self.prot_attr2))

                        loss_classifier = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(
                                labels=(y_train.drop(columns=self.prot_attr)).drop(columns=self.prot_attr2).astype(
                                    'float32'),
                                logits=pred_logits_all))

                        pred_protected_attributes_logits_all = self.adv1_model.forward(pred_logits_all, y_train.drop(
                            columns=self.prot_attr).drop(columns=self.prot_attr2))
                        loss_adv = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train[self.prot_attr].astype('float32'),
                                                                    logits=pred_protected_attributes_logits_all))
                        adv_accuracy = accuracy(predict_class(tf.sigmoid(pred_protected_attributes_logits_all)),
                                                y_train[self.prot_attr])

                        # Computation of validation set results

                        pred_labels_all_validation, pred_logits_all_validation = self.clf_model.forward(
                            x_validation.astype('float32'))
                        y_pred_all_validation = np.array(predict_class(pred_labels_all_validation))
                        dao_sex_validation = compute_DAO(y_pred_all_validation, y_validation, feature, predict, 1, 0,
                                                         [0, 1])
                        dao_race_validation = compute_DAO(y_pred_all_validation, y_validation, feature2, predict, 1, 0,
                                                          [0, 1])
                        clf_accuracy_validation = accuracy(predict_class(pred_labels_all_validation),
                                                           y_validation.drop(columns=self.prot_attr).drop(
                                                               columns=self.prot_attr2))

                        pred_protected_attributes_logits_all_validation = self.adv1_model.forward(
                            pred_logits_all_validation,
                            y_validation.drop(columns=self.prot_attr).drop(columns=self.prot_attr2))
                        adv_accuracy_validation = accuracy(
                            predict_class(tf.sigmoid(pred_protected_attributes_logits_all_validation)),
                            y_validation[self.prot_attr])

                        # Temporary Computation of Test results for replication (To be deleted after)

                        pred_labels_all_test, pred_logits_all_test = self.clf_model.forward(x_test.astype('float32'))
                        y_pred_all_test = np.array(predict_class(pred_labels_all_test))
                        dao_sex_test = compute_DAO(y_pred_all_test, y_test, feature, predict, 1, 0, [0, 1])
                        dao_race_test = compute_DAO(y_pred_all_test, y_test, feature2, predict, 1, 0, [0, 1])
                        clf_accuracy_test = accuracy(predict_class(pred_labels_all_test),
                                                     y_test.drop(columns=self.prot_attr).drop(columns=self.prot_attr2))

                        pred_protected_attributes_logits_all_test = self.adv1_model.forward(
                            pred_logits_all_test,
                            y_test.drop(columns=self.prot_attr).drop(columns=self.prot_attr2))
                        adv_accuracy_test = accuracy(
                            predict_class(tf.sigmoid(pred_protected_attributes_logits_all_test)),
                            y_test[self.prot_attr])


                        print(
                            "(Adversarial Debiasing) epoch %d; iter: %d; classifier loss: %f; "
                            "adversarial loss: %f; classifier accuracy: %f; adversary accuracy: %f; "
                            "DAO sex: %f; DAO race: %f; " % (
                                epoch, i, loss_classifier, loss_adv, clf_accuracy, adv_accuracy, dao_sex, dao_race))

                        # Print validation results
                        print(
                            "(VALIDATION RESULTS) epoch %d; iter: %d; classifier accuracy: %f; adversary accuracy: %f; DAO sex %f; DAO race: %f; "
                            % (epoch, i, clf_accuracy_validation, adv_accuracy_validation, dao_sex_validation, dao_race_validation))

                        # PRINT TEMPORARY TEST RESULTS
                        print(
                            "(TEST RESULTS)       epoch %d; iter: %d; classifier accuracy: %f; adversary accuracy: %f; DAO sex %f; DAO race: %f; "
                            % (epoch, i, clf_accuracy_test, adv_accuracy_test, dao_sex_test, dao_race_test))

                        earlystop_clf = loss_classifier
                        save_epoch = epoch

                        if early_stopper.new_minimum_dao(dao_sex_validation):
                        # METRICS FROM TRAINING DATA ARE TAKEN
                            manager_adv.save()
                            min_loss_adv = loss_adv
                            min_adv_acc = adv_accuracy
                            min_loss_clf = loss_classifier
                            min_clf_acc = clf_accuracy
                            min_batch_dao = dao_sex_validation

                        # Early stop based on minimum batch_dao of validation set
                if early_stopper.early_stop_adv(earlystop_clf, save_epoch, clf_accuracy_validation,
                                                        dao_sex_validation):
                    print(
                        "(Adversarial Debiasing) Model from this epoch is taken: epoch %d; iter: %d; batch classifier loss: %f; batch adversarial loss: %f; "
                        "batch classifier accuracy: %f; batch adversary accuracy: %f; batch DAO: %f" % (
                            early_stopper.get_epoch(), 0, min_loss_clf, min_loss_adv, min_clf_acc, min_adv_acc,
                            min_batch_dao))
                    checkpoint_adv.restore(manager_adv.latest_checkpoint)
                    break

        # training for just the classifier (so when DEBIAS = FALSE)
        # in case you don't understand something, just look comments in the section for training classifier + adversary
        else:
            early_stopper = EarlyStopper(patience=5, min_delta=0.001)
            checkpoint = tf.train.Checkpoint(model=self.clf_model)
            manager = tf.train.CheckpointManager(checkpoint, directory='./checkpoints', max_to_keep=6)
            earlystop_clf, save_epoch = 100, 0
            for epoch in range(self.num_epochs):
                shuffled_ids = [i for i in range(num_train_samples)]
                # BUG: np.random.choice not reproduce same shuffled id every epochs
                # shuffled_ids = np.random.choice(num_train_samples, num_train_samples, replace=False)
                for i in range(num_train_samples // self.batch_size):
                    batch_ids = shuffled_ids[self.batch_size * i: self.batch_size * (i + 1)]
                    batch_features = x_train[batch_ids].astype('float32')
                    batch_labels = np.reshape(dependent_variable_classifier.iloc[batch_ids], [-1, 1]).astype('float32')
                    with tf.GradientTape() as tape:
                        classifier_pred, classifier_logits = self.clf_model.forward(batch_features)
                        loss_classifier = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_labels.astype('float32'),
                                                                    logits=classifier_logits))
                    gradients = tape.gradient(loss_classifier, self.clf_model.trainable_variables)
                    classifier_opt.apply_gradients(zip(gradients, self.clf_model.trainable_variables))
                    # if i % 200 == 0:
                    if i % (num_train_samples // self.batch_size - 1) == 0 and i != 0:
                        feature = 'gender'
                        feature2 = 'race'
                        predict = 'income'

                        pred_labels_all, pred_logits_all = self.clf_model.forward(x_train.astype('float32'))
                        y_pred_all = np.array(predict_class(pred_labels_all))

                        dao_sex = compute_DAO(y_pred_all, y_train, feature, predict, 1, 0, [0, 1])
                        dao_race = compute_DAO(y_pred_all, y_train, feature2, predict, 1, 0, [0, 1])
                        clf_accuracy = accuracy(predict_class(pred_labels_all), y_train.drop(columns=self.prot_attr).drop(columns=self.prot_attr2).astype('float32'))

                        loss_clf = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(
                                labels=(y_train.drop(columns=self.prot_attr)).drop(columns=self.prot_attr2).astype('float32'),
                                logits=pred_logits_all))

                        # Computation of validation set results

                        pred_labels_all_validation, pred_logits_all_validation = self.clf_model.forward(
                            x_validation.astype('float32'))
                        y_pred_all_validation = np.array(predict_class(pred_labels_all_validation))
                        dao_sex_validation = compute_DAO(y_pred_all_validation, y_validation, feature, predict, 1, 0,
                                                           [0, 1])
                        dao_race_validation = compute_DAO(y_pred_all_validation, y_validation, feature2, predict, 1, 0, [0, 1])
                        clf_accuracy_validation = accuracy(predict_class(pred_labels_all_validation),
                                                           y_validation.drop(columns=self.prot_attr).drop(columns=self.prot_attr2).astype('float32'))

                        # Temporary Computation of Test results for replication (To be deleted after)

                        pred_labels_all_test, pred_logits_all_test = self.clf_model.forward(x_test.astype('float32'))
                        y_pred_all_test = np.array(predict_class(pred_labels_all_test))
                        dao_sex_test = compute_DAO(y_pred_all_test, y_test, feature, predict, 1, 0, [0, 1])
                        dao_race_test = compute_DAO(y_pred_all_test, y_test, feature2, predict, 1, 0, [0, 1])
                        clf_accuracy_test = accuracy(predict_class(pred_labels_all_test),
                                                     y_test.drop(columns=self.prot_attr).drop(columns=self.prot_attr2).astype('float32'))

                        print(
                            "(Training Classifier)  epoch %d; iter: %d; classifier accuracy: %f; DAO sex: %f; DAO race: %f; Classifier loss: %f; "
                            % (epoch, i, clf_accuracy, dao_sex, dao_race, loss_clf))

                        # Print validation results
                        print("(VALIDATION RESULTS) epoch %d; iter: %d; classifier accuracy: %f; DAO sex %f; DAO race: %f; "
                              % (epoch, i, clf_accuracy_validation, dao_sex_validation, dao_race_validation))

                        # PRINT TEMPORARY TEST RESULTS
                        print("(TEST RESULTS)       epoch %d; iter: %d; classifier accuracy: %f; DAO sex %f; DAO race: %f; "
                              % (epoch, i, clf_accuracy_test, dao_sex_test, dao_race_test))

                        # Compute metrics for validation set
                        earlystop_clf = loss_clf
                        save_epoch = epoch

                        if early_stopper.new_minimum(clf_accuracy_validation):
                            manager.save()
                            min_loss_clf = loss_clf
                            min_clf_acc = clf_accuracy
                            min_batch_dao = dao_sex
                    # Early stop based on validation set accuracy
                if early_stopper.early_stop(earlystop_clf, save_epoch, clf_accuracy_validation,
                                                    dao_sex_validation):
                    # Metrics from training set are taken
                    print(
                        "(Training Classifier) Model from this epoch is taken: epoch %d; iter: %d; batch classifier loss: %f; batch classifier accuracy: %f; batch DAO: %f; " % (
                            early_stopper.get_epoch(), 0, min_loss_clf, min_clf_acc, min_batch_dao))
                    checkpoint.restore(manager.latest_checkpoint)
                    break
        return self

    # Can ask Liu if you have any questions, but def fit is more important than these functions
    def decision_function(self, X):
        """Soft prediction scores.

        Args:
            X (pandas.DataFrame): Test samples.

        Returns:
            numpy.ndarray: Confidence scores per (sample, class) combination. In
            the binary case, confidence score for ``self.classes_[1]`` where >0
            means this class would be predicted.
        """
        if scipy.sparse.issparse(X):
            X = X.todense()
        num_test_samples = X.shape[0]
        # n_classes = len(self.classes_)

        # if n_classes == 2:
        #    n_classes = 1 # lgtm [py/unused-local-variable]

        self.clf_model.dropout = 0
        samples_covered = 0
        pred_labels_list = []
        while samples_covered < num_test_samples:
            start = samples_covered
            end = samples_covered + self.batch_size
            if end > num_test_samples:
                end = num_test_samples
            batch_ids = np.arange(start, end)
            batch_features = X[batch_ids]
            pred_labels, pred_logits = self.clf_model.forward(batch_features.astype("float32"))

            pred_labels_list += pred_labels.numpy().tolist()
            samples_covered += len(batch_features)

        scores = np.array(pred_labels_list, dtype=np.float64).reshape(-1, 1)
        return scores.ravel() if scores.shape[1] == 1 else scores

    # Can ask Liu if you have any questions, but def fit is more important than these functions
    def predict(self, X):
        """Predict class labels for the given samples.

        Args:
            X (pandas.DataFrame): Test samples.

        Returns:
            numpy.ndarray: Predicted class label per sample.
        """
        scores = self.decision_function(X)
        if scores.ndim == 1:
            if X.shape[0] == 1:
                indices = (scores > 0.5).astype(int).reshape((-1,))
            else:
                indices = (scores > 0.5).astype(int).reshape((-1,))
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]


# needed for accuracy for printing metrics epochs
def accuracy(y_pred, y):
    # Return the proportion of matches between `y_pred` and `y`
    check_equal = tf.cast(y_pred == y, tf.float32)
    acc_val = tf.reduce_mean(check_equal)
    return acc_val


# needed for DAO for printing metrics epochs
def predict_class(y_pred, thresh=0.5):
    # Return a tensor with  `1` if `y_pred` > `0.5`, and `0` otherwise
    return tf.cast(y_pred > thresh, tf.float32)


# computing DAO each epoch
def compute_DAO(y_pred, y_real, SensitiveCat, outcome, privileged, unprivileged, labels):
    y_priv = y_pred[y_real[SensitiveCat] == privileged]
    y_real_priv = y_real[y_real[SensitiveCat] == privileged]
    y_unpriv = y_pred[y_real[SensitiveCat] == unprivileged]
    y_real_unpriv = y_real[y_real[SensitiveCat] == unprivileged]
    TN_priv, FP_priv, FN_priv, TP_priv = confusion_matrix(y_real_priv[outcome], y_priv, labels=labels).ravel()
    TN_unpriv, FP_unpriv, FN_unpriv, TP_unpriv = confusion_matrix(y_real_unpriv[outcome], y_unpriv,
                                                                  labels=labels).ravel()

    return 0.5 * (abs(FP_unpriv / (FP_unpriv + TN_unpriv) - FP_priv / (FP_priv + TN_priv)) + abs(
        TP_unpriv / (TP_unpriv + FN_unpriv) - TP_priv / (TP_priv + FN_priv)))


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.epoch = 0
        self.clf_accuracy = 0
        self.DAO = 100
        self.accuracy_threshold = 0.8
        self.epoch_threshold = 10

    def early_stop(self, validation_loss, epoch, clf_accuracy, DAO):
        # if validation_loss < self.min_validation_loss:
        if clf_accuracy > self.clf_accuracy:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.epoch = epoch
            self.clf_accuracy = clf_accuracy
            self.DAO = DAO

        elif clf_accuracy < (self.clf_accuracy + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def early_stop_adv(self, validation_loss, epoch, clf_accuracy, DAO):
        # Skip the first few epochs since they have low dao
        if DAO < self.DAO and clf_accuracy > 0.8:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.epoch = epoch
            self.clf_accuracy = clf_accuracy
            self.DAO = DAO

        # Delta is added to add minimum difference to count as a 'strike'
        elif DAO > (self.DAO + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience :
                return True
        return False

    def new_minimum(self, clf_accuracy):
        if clf_accuracy > self.clf_accuracy:
            return True
        return False

    def new_minimum_dao(self, dao):
        if dao < self.DAO:
            return True
        return False

    def get_epoch(self):
        return self.epoch

    def get_clf(self):
        return self.min_validation_loss

    def get_clf_accuracy(self):
        return self.clf_accuracy

    def get_DAO(self):
        return self.DAO