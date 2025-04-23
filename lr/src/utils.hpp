#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <Eigen/Dense>
#include <stdexcept>
#include <random>
#include <map>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>





class ForwardLR {

    public:

        Eigen::VectorXd forward(Eigen::MatrixXd X, Eigen::MatrixXd W, Eigen::VectorXd b) {

            if (X.cols() != W.rows() || W.cols() != 1 || b.size() != 1) {
                throw std::runtime_error("Shape mismatch: X.cols() must equal W.rows(), W.cols() must equal 1, and b.size() must equal 1");
            }

            return (X * W).col(0) + Eigen::VectorXd::Constant(X.rows(), b(0));
        }
};



class DForwardLR {

    public:

        std::map<std::string, Eigen::VectorXd> backward(Eigen::VectorXd grad_y_hat, Eigen::MatrixXd X) {

            if (grad_y_hat.size() != X.rows()) {
                throw std::runtime_error("Shape mismatch: grad_y_hat.size() must equal X.rows()");
            }

            std::map<std::string, Eigen::VectorXd> grads;
            Eigen::VectorXd dW = (X.transpose() * grad_y_hat) / X.rows();
            Eigen::VectorXd db(1);

            db(0) = grad_y_hat.sum() / X.rows();
            grads["dW"] = dW;
            grads["db"] = db;

            return grads;
        }
};



class MSE {

    public:

        double mean_squared_error(Eigen::VectorXd y_hat, Eigen::VectorXd y) {

            if (y_hat.size() != y.size()) {
                throw std::runtime_error("y_hat.size() must equal y.size()");
            }

            if (y.size() == 0) {
                throw std::runtime_error("Input vectors cannot be empty");
            }

            Eigen::VectorXd resids = y - y_hat;
            double mse = resids.array().square().sum() / y.size();

            return mse;
        }
};



class DMSE {

    public:

        Eigen::VectorXd dmean_squared_error(Eigen::VectorXd y_hat, Eigen::VectorXd y) {

            if (y_hat.size() != y.size()) {
                throw std::runtime_error("y_hat.size() must equal y.size()");
            }

            if (y.size() == 0) {
                throw std::runtime_error("Input vectors cannot be empty");
            }

            Eigen::VectorXd grad_y_hat = (2.0 / y.size()) * (y_hat - y);

            return grad_y_hat;
        }
};



class Params {

    public:

        std::map<std::string, Eigen::VectorXd> init_params(int num_feat, double scaling_factor = 1.0, unsigned int seed = 0) {

            if (num_feat <= 0) {
                throw std::runtime_error("Number of features must be positive");
            }

            std::map<std::string, Eigen::VectorXd> params;
            Eigen::VectorXd ws(num_feat);
            std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
            std::normal_distribution<> dist(0.0, 0.01); // normal distribution for ML

            for (int i = 0; i < num_feat; ++i) {
                ws[i] = dist(gen) * scaling_factor;
            }

            Eigen::VectorXd b(1);
            b[0] = dist(gen) * scaling_factor;
            params["W"] = ws;
            params["b"] = b;

            return params;
        }

        void update(std::map<std::string, Eigen::VectorXd>& params, std::map<std::string, Eigen::VectorXd> grads, double lr) {

            params["W"] -= lr * grads["dW"];
            params["b"] -= lr * grads["db"];
        }
};



class Train {

    public:

        std::vector<double> losses;
        std::map<std::string, Eigen::VectorXd> params;

        void trainer(Eigen::MatrixXd X, Eigen::VectorXd y, int num_iters = 1000, double lr = 0.01, int loss_freq = 100) {

            if (X.rows() != y.size()) {
                throw std::runtime_error("Shape mismatch: y.size() must equal X.rows()");
            }

            if (num_iters <= 0) {
                throw std::runtime_error("Number of training iterations must be positive");
            }

            // initialize classes
            ForwardLR ford;
            DForwardLR bacd;
            MSE mse;
            DMSE dmse;
            Params par;

            // initialize parameters
            params = par.init_params(X.cols(), 0.2, 42);

            Eigen::VectorXd y_hat;
            Eigen::VectorXd grad_y_hat;
            double loss;
            std::map<std::string, Eigen::VectorXd> grads;

            for (int i = 0; i < num_iters; ++i) {

                // forward propagation
                y_hat = ford.forward(X, params["W"], params["b"]);
                loss = mse.mean_squared_error(y_hat, y);

                // backward propagation
                grad_y_hat = dmse.dmean_squared_error(y_hat, y);
                grads = bacd.backward(grad_y_hat, X);

                // update parameters
                par.update(params, grads, lr);
                losses.push_back(loss);

                // print every loss_freq iterations or last iteration
                if (i % loss_freq == 0 || i == num_iters - 1) {
                    std::cout << "Iteration: " << i << " | Loss: " << loss << std::endl;
                }
            }
        }
};




#endif 