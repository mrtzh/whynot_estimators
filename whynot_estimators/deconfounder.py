"""Deconfounded ordinary least-squares based-estimators for causal inference."""
from time import perf_counter

# pylint:disable-msg=invalid-name
import numpy as np
from whynot.framework import InferenceResult

import whynot_estimators


class Deconfounder(whynot_estimators.Estimator):
    """Run deconfounded ols to estimate causal effect of treatment on the outcome y.

    Models the outcome variable Y as:
        Y = b_0 + b_1 * treatment + b_C^T * control.

    For more information on the deconfounder, please refer to:
        Wang, Yixin, and David M. Blei. "The blessings of multiple causes." arXiv
        preprint arXiv:1805.06826 (2018).

    """

    @property
    def name(self):
        """Estimator name."""
        return "deconfounder"

    def import_estimator(self):
        """Verify tensorflow dependencies are installed."""
        # pylint:disable-msg=unused-import
        import statsmodels.api
        import tensorflow
        from tensorflow_probability import edward2

    # pylint:disable-msg=too-many-locals,too-many-statements
    def estimate_treatment_effect(self, covariates, treatment, outcome):
        """Run deconfounded ols to estimate causal effect of treatment on the outcome y.

        Models the outcome variable Y as:
            Y = b_0 + b_1 * treatment + b_C^T * control.

        For more information on the deconfounder, please refer to:
            Wang, Yixin, and David M. Blei. "The blessings of multiple causes." arXiv
            preprint arXiv:1805.06826 (2018).

        Parameters
        ----------
            covariates: `np.ndarray`
                Array of shape [num_samples, num_features] of features
            treatment:  `np.ndarray`
                Array of shape [num_samples]  indicating treatment status for each
                sample.
            outcome:  `np.ndarray`
                Array of shape [num_samples] containing the observed outcome for
                each sample.

        Returns
        -------
            result: `whynot.framework.InferenceResult`
                InferenceResult object for this procedure

        """
        if not self.is_installed():
            raise ValueError(f"Estimator {self.name} is not installed!")

        import statsmodels.api as sm
        import tensorflow as tf
        from tensorflow_probability import edward2 as ed

        features = np.copy(covariates)
        treatment = treatment.reshape(-1, 1)
        features = np.concatenate([treatment, features], axis=1)

        def ppca_model(data_dim, latent_dim, num_datapoints, stddv_datapoints):
            """Return probabilistic pca model."""
            # pylint:disable-msg=no-member
            w = ed.Normal(
                loc=tf.zeros([latent_dim, data_dim]),
                scale=tf.ones([latent_dim, data_dim]),
                name="w",
            )  # parameter
            z = ed.Normal(
                loc=tf.zeros([num_datapoints, latent_dim]),
                scale=tf.ones([num_datapoints, latent_dim]),
                name="z",
            )  # local latent variable / substitute confounder
            x = ed.Normal(
                loc=tf.multiply(tf.matmul(z, w), 1),
                scale=stddv_datapoints * tf.ones([num_datapoints, data_dim]),
                name="x",
            )  # (modeled) data
            return x, (w, z)

        def variational_model(qw_mean, qw_stddv, qz_mean, qz_stddv):
            """Return variational model."""
            # pylint:disable-msg=no-member
            qw = ed.Normal(loc=qw_mean, scale=qw_stddv, name="qw")
            qz = ed.Normal(loc=qz_mean, scale=qz_stddv, name="qz")
            return qw, qz

        start_time = perf_counter()

        # do probabilistic PCA
        # pylint:disable-msg=unpacking-non-sequence
        log_joint = ed.make_log_joint_fn(ppca_model)
        latent_dim = 2
        stddv_datapoints = 0.1
        num_datapoints, data_dim = features.shape

        model = ppca_model(
            data_dim=data_dim,
            latent_dim=latent_dim,
            num_datapoints=num_datapoints,
            stddv_datapoints=stddv_datapoints,
        )

        def target(w, z):
            """Unnormalized target density as a function of the parameters."""
            return log_joint(
                data_dim=data_dim,
                latent_dim=latent_dim,
                num_datapoints=num_datapoints,
                stddv_datapoints=stddv_datapoints,
                w=w,
                z=z,
                x=features,
            )

        log_q = ed.make_log_joint_fn(variational_model)

        def target_q(qw, qz):
            return log_q(
                qw_mean=qw_mean,
                qw_stddv=qw_stddv,
                qz_mean=qz_mean,
                qz_stddv=qz_stddv,
                qw=qw,
                qz=qz,
            )

        qw_mean = tf.Variable(np.ones([latent_dim, data_dim]), dtype=tf.float32)
        qz_mean = tf.Variable(np.ones([num_datapoints, latent_dim]), dtype=tf.float32)
        qw_stddv = tf.nn.softplus(
            tf.Variable(-4 * np.ones([latent_dim, data_dim]), dtype=tf.float32)
        )
        qz_stddv = tf.nn.softplus(
            tf.Variable(-4 * np.ones([num_datapoints, latent_dim]), dtype=tf.float32)
        )

        qw, qz = variational_model(
            qw_mean=qw_mean, qw_stddv=qw_stddv, qz_mean=qz_mean, qz_stddv=qz_stddv
        )

        energy = target(qw, qz)
        entropy = -target_q(qw, qz)
        elbo = energy + entropy

        optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
        train = optimizer.minimize(-elbo)

        init = tf.global_variables_initializer()

        t = []
        num_epochs = 500
        with tf.Session() as sess:
            sess.run(init)

            for i in range(num_epochs):
                sess.run(train)
                if i % 5 == 0:
                    t.append(sess.run([elbo]))

                z_mean_inferred = sess.run(qz_mean)

        # approximate the (random variable) substitute confounders with their inferred mean.
        Z_hat = z_mean_inferred
        # augment the regressors to be both the covariates and the substitute confounder Z
        features_aug = np.column_stack([features, Z_hat])

        # do standard OLS on augmented features
        features_aug = sm.add_constant(features_aug, prepend=True, has_constant="add")

        model = sm.OLS(outcome, features_aug)
        results = model.fit()

        stop_time = perf_counter()

        # Treatment is the second variable (first is the constant offset)
        ate = results.params[1]
        stderr = results.bse[1]
        conf_int = (ate - 1.96 * stderr, ate + 1.96 * stderr)
        return InferenceResult(
            ate=ate,
            stderr=stderr,
            ci=conf_int,
            individual_effects=None,
            elapsed_time=stop_time - start_time,
        )


DECONFOUNDER = Deconfounder()
