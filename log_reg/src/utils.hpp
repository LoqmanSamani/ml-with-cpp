#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <Eigen/Dense>
#include <map>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <random>
#include <numeric> 
#include "data_proc.hpp"







class ComputeZ{

    public:

        Eigen::VectorXd forward(const Eigen::MatrixXd& X, const Eigen::VectorXd& w, const Eigen::VectorXd& b) const {

            if (X.size() == 0 || w.size() == 0 || b.size() == 0){

                throw std::runtime_error("At least one input is empty! X.size(): " + 
                      std::to_string(X.size())+ " w.size(): " +
                      std::to_string(w.size()) + " b.size(): " + 
                      std::to_string(b.size()));
            }

            
            if (X.cols() != w.rows() || w.cols() != 1 || b.size() != 1){

                throw std::runtime_error("Shape Mismatch: X.cols(): " +
                      std::to_string(X.cols()) + " w.rows(): " +
                      std::to_string(w.rows()) + " b.size(): " +
                      std::to_string(b.size()));
            }
            
            return X * w + Eigen::VectorXd::Constant(X.rows(), b(0));
        }

        std::map<std::string, Eigen::VectorXd> backward(const Eigen::VectorXd& z_grad, const Eigen::MatrixXd& X) const {

            if (z_grad.size() == 0 || X.rows() == 0){

                throw std::runtime_error("At least one input is empty! z_grad.size(): " +
                      std::to_string(z_grad.size()) + " X.size(): " +
                      std::to_string(X.size()));
            }

            if (X.rows() != z_grad.size()){

                throw std::runtime_error("Shape Mismatch: X.rows(): " + 
                      std::to_string(X.rows()) + " z_grad.size(): " + 
                      std::to_string(z_grad.size()));
            }

            std::map <std::string, Eigen::VectorXd> grads;
            Eigen::VectorXd dw = (X.transpose() * z_grad) / X.rows();
            Eigen::VectorXd db(1);

            db(0) = z_grad.sum() / X.rows();
            grads["dw"] = dw;
            grads["db"] = db;

            return grads;
        }
};


class Sigmoid {

    public:

      Eigen::VectorXd forward(const Eigen::VectorXd& z) const {

        return 1.0 / (1.0 + (-z.array()).exp());
      }
      
      Eigen::VectorXd backward(const Eigen::VectorXd& dy_hat) const {

        return dy_hat.array() * (1.0 - dy_hat.array());
      }
};


class LogLoss {

    public:

        LogLoss(double lambda = 0.0) : lambda(lambda) {}

        double forward(const Eigen::VectorXd& y, const Eigen::VectorXd& y_hat, const Eigen::VectorXd& w) const {

            if (y.size() != y_hat.size()) {
            throw std::runtime_error("Shape Mismatch: y.size(): " +
                std::to_string(y.size()) + " != y_hat.size(): " +
                std::to_string(y_hat.size()));
            }

            // l2 regularization term
            double l2 = w.squaredNorm() * (lambda/2);
            
            // binary cross entropy loss
            Eigen::VectorXd l = -(y.array() * y_hat.array().log() + (1.0 - y.array()) * (1.0 - y_hat.array()).log());
            
            return l.sum() / y.size() + l2;
        }
        
        Eigen::VectorXd backward(const Eigen::MatrixXd& X, const Eigen::VectorXd& y_hat, const Eigen::VectorXd& y) const {

            if (X.rows() != y_hat.size() || y_hat.size() != y.size()) {

            throw std::runtime_error("Shape Mismatch: X.rows(): " + 
                    std::to_string(X.rows()) + " y.size(): " + 
                    std::to_string(y.size()) + " y_hat.size(): " +
                    std::to_string(y_hat.size()));
            }

            if (X.size() == 0 || y.size() == 0 || y_hat.size() == 0) {

            throw std::runtime_error("At least one input is empty! X.size(): " + 
                    std::to_string(X.size()) + " y.size(): " + 
                    std::to_string(y.size()) + " y_hat.size(): " +
                    std::to_string(y_hat.size()));
            }

            //Eigen::VectorXd dl2 = lambda_ * w.array();
            
            return (y_hat - y)  / y.size();
        }

    private:
        double lambda;
};


class Params {

    public:

        Params(int num_feats = 1, double scaling_factor = 0.0, int seed = 42):
            num_feats(num_feats), scaling_factor(scaling_factor), seed(seed) {}

        void init_params(std::map<std::string, Eigen::VectorXd>& params) {

            if (num_feats <= 0) {
                throw std::runtime_error("Number of features must be positive: " + std::to_string(num_feats));
            }

            Eigen::VectorXd ws(num_feats);
            std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
            std::normal_distribution<> dist(0.0, 0.01);

            for (int i = 0; i < num_feats; ++i) {
                ws(i) = dist(gen) * scaling_factor;
            }

            Eigen::VectorXd b = Eigen::VectorXd::Constant(1, dist(gen) * scaling_factor);
            params["w"] = ws;
            params["b"] = b;
        }

        void update(std::map<std::string, Eigen::VectorXd>& params, const std::map<std::string, Eigen::VectorXd>& grads, double lr) {

            params["w"].noalias() -= lr * grads.at("dw");
            params["b"].noalias() -= lr * grads.at("db");
        }

    private:
        int num_feats;
        double scaling_factor;
        int seed;
};




class Train {

    public:
        
        Train(
            int epochs = 1000, double lr = 0.01, int val_freq = 100, int loss_freq = 100, int train_batch_size = 100,
            int val_batch_size = 50, bool use_val = false, double lambda_ = 0.01, double scaling_factor = 0.2, bool normalize = true, unsigned int seed = 42): 
            epochs(epochs), lr(lr), val_freq(val_freq), train_batch_size(train_batch_size), val_batch_size(val_batch_size), 
            loss_freq(loss_freq), use_val(use_val), lambda_(lambda_), scaling_factor(scaling_factor), normalize(normalize), seed(seed) {}

        void trainer(Eigen::MatrixXd& train_data, Eigen::MatrixXd& val_data, Eigen::MatrixXd& test_data, bool test = false) {

            if (train_batch_size > train_data.rows()) {

                throw std::runtime_error("train_batch_size: " +
                      std::to_string(train_batch_size) + ", must be less or equal to train_data.rows(): " +
                      std::to_string(train_data.rows()));
            }

            if (use_val && val_batch_size > val_data.rows()){

                throw std::runtime_error("val_batch_size: " +
                      std::to_string(val_batch_size) + ", must be less or equal to val_data.rows(): " +
                      std::to_string(val_data.rows()));
            }

            if (epochs <= 0){ 

                throw std::runtime_error("Number of epochs must be positive: " + std::to_string(epochs));
            }
            // normalize data
            if (normalize){
                standardize(train_data);
                if (use_val){
                    standardize(val_data);
                }
                if (test){
                    standardize(test_data);
                }
            }
            

            // initialize used classes
            DataProcessor dp;
            ComputeZ ford;
            Sigmoid act;
            Params par(train_data.cols()-1, scaling_factor, seed);
            LogLoss cost(lambda_);

            // initialize parameters and losses
            par.init_params(params);
            losses.reserve(epochs);
            //val_losses.reserve(epochs / val_freq + 1);

            // calculate number of iteration in each epoch
            int num_train_iters = (train_data.rows() + train_batch_size - 1) / train_batch_size;
            int num_val_iters = 0;
            if(use_val){
                num_val_iters = (val_data.rows() + val_batch_size - 1) / val_batch_size;
            }

            // create index vector for shuffling
            std::vector<int> train_indices(train_data.rows());
            std::iota(train_indices.begin(), train_indices.end(), 0);
            std::mt19937 g(seed);
            
            std::vector<int> val_indices;
            if (use_val) {
                val_indices.resize(val_data.rows());
                std::iota(val_indices.begin(), val_indices.end(), 0);
            }
            std::mt19937 r(seed);
            
            for (int epoch = 0; epoch < epochs; epoch++){
                double epoch_loss = 0.0;
                double val_loss = 0.0;

                // shuffle indices
                std::shuffle(train_indices.begin(), train_indices.end(), g);

                int start = 0; 
                int end = std::min(train_batch_size, static_cast<int>(train_data.rows()));
                
                for (int iter = 0; iter < num_train_iters; iter++){

                    int batch_size = end - start;
                    Eigen::MatrixXd X_batch(batch_size, train_data.cols()-1);
                    Eigen::VectorXd y_batch(batch_size);
                   
                    // slicing of indices
                    auto first = train_indices.begin() + start;
                    auto last = train_indices.begin() + end;
                    std::vector<int> indices_(first, last);
                    
                    dp.prepare_batch(train_data, X_batch, y_batch, indices_);

                    // forward propagation
                    Eigen::VectorXd z = ford.forward(X_batch, params["w"], params["b"]);
                    Eigen::VectorXd y_hat = act.forward(z);
                    double loss = cost.forward(y_batch, y_hat, params["w"]);

                    // backward propagation
                    Eigen::VectorXd dy_hat = cost.backward(X_batch, y_hat, y_batch); 
                    Eigen::VectorXd dz = act.backward(y_hat).array() * dy_hat.array();
                    std::map<std::string, Eigen::VectorXd> grads = ford.backward(dz, X_batch);

                    
                    // add l2 regularization gradient to dw
                    grads["dw"] += lambda_ * params["w"];
                    
                    // update parameters
                    par.update(params, grads, lr);

                    epoch_loss += loss;
                
                    start = end;
                    end = std::min(end + train_batch_size, static_cast<int>(train_data.rows()));
                }
                
                double mean_loss = epoch_loss / num_train_iters;
                losses.push_back(mean_loss);
                
                // validation
                if (use_val && epoch % val_freq == 0){

                    std::shuffle(val_indices.begin(), val_indices.end(), r);

                    int start = 0; 
                    int end = std::min(val_batch_size, static_cast<int>(val_data.rows()));
                    
                    for (int iter = 0; iter < num_val_iters; iter++){

                        int batch_size = end - start;
                        Eigen::MatrixXd X_batch(batch_size, val_data.cols()-1);
                        Eigen::VectorXd y_batch(batch_size);
                    
                        auto first = val_indices.begin() + start;
                        auto last = val_indices.begin() + end;
                        std::vector<int> indices_(first, last);
                        
                        dp.prepare_batch(val_data, X_batch, y_batch, indices_);

                        Eigen::VectorXd z = ford.forward(X_batch, params["w"], params["b"]);
                        Eigen::VectorXd y_hat = act.forward(z);
                        val_loss += cost.forward(y_batch, y_hat, params["w"]);

                        start = end;
                        end = std::min(end + val_batch_size, static_cast<int>(val_data.rows()));
                    }

                    val_losses.push_back(val_loss / num_val_iters);
                }

                if (epoch % loss_freq == 0){

                    std::cout << "Epoch: " << epoch << " | Loss: " << mean_loss << std::endl;

                    if (use_val && epoch % val_freq == 0){
                        std::cout << "Val Loss: " << val_loss / num_val_iters << std::endl;
                    }
                }
            }

            if (test) {

                Eigen::MatrixXd X_test(test_data.rows(), test_data.cols() - 1);
                Eigen::VectorXd y_test(test_data.rows());
                std::vector<int> test_indices(test_data.rows());
                std::iota(test_indices.begin(), test_indices.end(), 0); 

                dp.prepare_batch(test_data, X_test, y_test, test_indices); 

                Eigen::VectorXd z_test = ford.forward(X_test, params["w"], params["b"]);
                Eigen::VectorXd y_hat_test = act.forward(z_test);

                double test_loss = cost.forward(y_test, y_hat_test, params["w"]);

                std::cout << "Test Loss: " << test_loss << std::endl;
            }
        }

        // getters for losses and parameters
        const std::vector<double>& get_losses() const {
            return losses;
        }

        const std::vector<double>& get_val_losses() const {
            return val_losses;
        }

        const std::map<std::string, Eigen::VectorXd>& get_params() const {
            return params;
        }

    private:

        double scaling_factor;
        unsigned int seed;
        bool use_val;
        int epochs;
        double lr;
        int val_freq;
        int train_batch_size;
        int val_batch_size; 
        int loss_freq;
        bool normalize;
        double lambda_;

        std::vector<double> losses;
        std::vector<double> val_losses;
        std::map<std::string, Eigen::VectorXd> params;
};


#endif 