#ifndef UTILS_HPP
#define UTILS_HPP

#include <Eigen/Dense>
#include <stdexcept>
#include <random>
#include <map>
#include <vector>
#include <algorithm>
#include <numeric>



class ForwardLR {

    public:

        Eigen::VectorXd forward(const Eigen::MatrixXd& X, const Eigen::MatrixXd& W, const Eigen::VectorXd& b) const {

            if (X.cols() != W.rows() || W.cols() != 1 || b.size() != 1) {

                throw std::runtime_error("Shape mismatch: X.cols()=" + std::to_string(X.cols()) +
                                        ", W.rows()=" + std::to_string(W.rows()) +
                                        ", W.cols()=" + std::to_string(W.cols()) +
                                        ", b.size()=" + std::to_string(b.size()));
            }

            return X * W + Eigen::VectorXd::Constant(X.rows(), b(0));
        }
};



class DForwardLR {

    public:

        std::map<std::string, Eigen::VectorXd> backward(const Eigen::VectorXd& grad_y_hat, const Eigen::MatrixXd& X) const {

            if (grad_y_hat.size() != X.rows()) {

                throw std::runtime_error("Shape mismatch: grad_y_hat.size()=" + std::to_string(grad_y_hat.size()) +
                                        ", X.rows()=" + std::to_string(X.rows()));
            }

            std::map<std::string, Eigen::VectorXd> grads;
            Eigen::VectorXd dW = X.transpose() * grad_y_hat / X.rows();
            Eigen::VectorXd db(1);
            db(0) = grad_y_hat.sum() / X.rows();
            grads["dW"] = dW;
            grads["db"] = db;

            return grads;
        }
};



class MSE {

    public:

        double mean_squared_error(const Eigen::VectorXd& y_hat, const Eigen::VectorXd& y) const {

            if (y_hat.size() != y.size()) {

                throw std::runtime_error("y_hat.size()=" + std::to_string(y_hat.size()) +
                                        " != y.size()=" + std::to_string(y.size()));
            }

            if (y.size() == 0) {
                throw std::runtime_error("Input vectors cannot be empty");
            }

            return (y - y_hat).squaredNorm() / y.size();
        }
};



class DMSE {

    public:

        Eigen::VectorXd dmean_squared_error(const Eigen::VectorXd& y_hat, const Eigen::VectorXd& y) const {

            if (y_hat.size() != y.size()) {

                throw std::runtime_error("y_hat.size()=" + std::to_string(y_hat.size()) +
                                        " != y.size()=" + std::to_string(y.size()));
            }

            if (y.size() == 0) {
                throw std::runtime_error("Input vectors cannot be empty");
            }

            return 2.0 / y.size() * (y_hat - y);
        }
};



class Params {

    public:
        std::map<std::string, Eigen::VectorXd> init_params(int num_feat, double scaling_factor = 1.0, unsigned int seed = 0) const {

            if (num_feat <= 0) {
                throw std::runtime_error("Number of features must be positive: " + std::to_string(num_feat));
            }

            std::map<std::string, Eigen::VectorXd> params;
            Eigen::VectorXd ws(num_feat);
            std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
            std::normal_distribution<> dist(0.0, 0.01);

            for (int i = 0; i < num_feat; ++i) {
                ws(i) = dist(gen) * scaling_factor;
            }

            Eigen::VectorXd b = Eigen::VectorXd::Constant(1, dist(gen) * scaling_factor);
            params["W"] = ws;
            params["b"] = b;

            return params;
        }

        void update(std::map<std::string, Eigen::VectorXd>& params, const std::map<std::string, Eigen::VectorXd>& grads, double lr) {

            params["W"].noalias() -= lr * grads.at("dW");
            params["b"].noalias() -= lr * grads.at("db");
        }
};



class Train {

    public:

        std::vector<double> losses;
        std::map<std::string, Eigen::VectorXd> params;
    
        Train(double scaling_factor = 0.2, unsigned int seed = 42)
            : scaling_factor_(scaling_factor), seed_(seed) {}
    
        void trainer(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, int epochs = 1000, int batch_size = 100, double lr = 0.01, int loss_freq = 100) {

            if (batch_size > X.rows()) {
                throw std::runtime_error("Batch size (" + std::to_string(batch_size) +
                                         ") must be <= number of samples (" + std::to_string(X.rows()) + ")");
            }

            if (X.rows() != y.size()) {
                throw std::runtime_error("Shape mismatch: X.rows()=" + std::to_string(X.rows()) +
                                         ", y.size()=" + std::to_string(y.size()));
            }

            if (epochs <= 0) {
                throw std::runtime_error("Number of epochs must be positive: " + std::to_string(epochs));
            }
    
            ForwardLR ford;
            DForwardLR bacd;
            MSE mse;
            DMSE dmse;
            Params par;
    
            params = par.init_params(X.cols(), scaling_factor_, seed_);
            losses.reserve(epochs);
    
            // create index vector for shuffling
            std::vector<int> indices(X.rows());
            std::iota(indices.begin(), indices.end(), 0);
            std::mt19937 g(seed_);
    
            for (int i = 0; i < epochs; ++i) {
                // shuffle indices
                std::shuffle(indices.begin(), indices.end(), g);
    
                double epoch_loss = 0.0;
                int num_iters = (X.rows() + batch_size - 1) / batch_size;
    
                for (int j = 0; j < num_iters; ++j) {

                    int start = j * batch_size;
                    int current_batch_size = std::min(batch_size, static_cast<int>(X.rows()) - start);
    
                    // extract batch using indices
                    Eigen::MatrixXd X_batch(current_batch_size, X.cols());
                    Eigen::VectorXd y_batch(current_batch_size);
                    
                    for (int k = 0; k < current_batch_size; ++k) {
                        int idx = indices[start + k];
                        X_batch.row(k) = X.row(idx);
                        y_batch(k) = y(idx);
                    }
    
                    Eigen::VectorXd y_hat_batch(current_batch_size);
                    y_hat_batch.noalias() = ford.forward(X_batch, params["W"], params["b"]);
                    double loss = mse.mean_squared_error(y_hat_batch, y_batch);
                    Eigen::VectorXd grad_y_hat_batch = dmse.dmean_squared_error(y_hat_batch, y_batch);
                    auto grads = bacd.backward(grad_y_hat_batch, X_batch);
                    par.update(params, grads, lr);
                    epoch_loss += loss;
                }
    
                double mean_epoch_loss = epoch_loss / num_iters;
                losses.push_back(mean_epoch_loss);
    
                if (i % loss_freq == 0 || i == epochs - 1) {
                    std::cout << "Epoch: " << i << " | Loss: " << mean_epoch_loss << std::endl;
                }
            }
        }
    
    private:
        double scaling_factor_;
        unsigned int seed_;
    };
    


#endif // UTILS_HPP