#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    // Loop over minibatches in order
    for (size_t start = 0; start < m; start += batch) {
        size_t end = std::min(start + batch, m);
        size_t bs = end - start;

        // 1) Compute logits for this batch: Z[b, j] = X[b]·theta[:, j]
        std::vector<float> logits(bs * k);
        for (size_t b = 0; b < bs; ++b) {
            const float *x_row = X + (start + b) * n;
            for (size_t j = 0; j < k; ++j) {
                float sum = 0.0f;
                for (size_t d = 0; d < n; ++d) {
                    sum += x_row[d] * theta[d * k + j];
                }
                logits[b * k + j] = sum;
            }
        }

        // 2) Softmax probabilities with numeric stabilization
        std::vector<float> probs(bs * k);
        for (size_t b = 0; b < bs; ++b) {
            // find max over classes
            float max_z = logits[b * k];
            for (size_t j = 1; j < k; ++j) {
                max_z = std::max(max_z, logits[b * k + j]);
            }
            // compute exp and sum
            float sum_exp = 0.0f;
            for (size_t j = 0; j < k; ++j) {
                float v = std::exp(logits[b * k + j] - max_z);
                probs[b * k + j] = v;
                sum_exp += v;
            }
            // normalize
            for (size_t j = 0; j < k; ++j) {
                probs[b * k + j] /= sum_exp;
            }
        }

        // 3) Compute gradient = X_batch^T · (probs - one_hot) / bs
        std::vector<float> grad(n * k, 0.0f);
        for (size_t b = 0; b < bs; ++b) {
            unsigned char yi = y[start + b];
            const float *x_row = X + (start + b) * n;
            for (size_t d = 0; d < n; ++d) {
                float xd = x_row[d];
                for (size_t j = 0; j < k; ++j) {
                    float indicator = (j == yi) ? 1.0f : 0.0f;
                    grad[d * k + j] += xd * (probs[b * k + j] - indicator);
                }
            }
        }
        float scale = lr / static_cast<float>(bs);

        // 4) Update theta in place
        for (size_t d = 0; d < n; ++d) {
            for (size_t j = 0; j < k; ++j) {
                theta[d * k + j] -= scale * grad[d * k + j];
            }
        }
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
