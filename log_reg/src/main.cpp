#include <iostream>
#include <Eigen/Dense>
#include <map>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <stdexcept>
#include <random>








class ComputeZ{

    public:

        Eigen::VectorXd forward(const Eigen::MatrixXd& X, const Eigen::VectorXd& w, const Eigen::VectorXd& b)const{

            if (X.size() == 0 || w.size() == 0 || b.size() == 0){
                throw std::runtime_error("At least one input is empty! X.size(): " + 
                      std::to_string(X.size())+ " w.size(): " +
                      std::to_string(w.size()) + " b.size(): " + 
                      std::to_string(b.size()));
            }

            if (X.cols() != w.rows() || w.cols() == 1 || b.size() == 1){
                throw std::runtime_error("Shape Missmatch: X.cols(): " +
                      std::to_string(X.cols()) + " w.rows():" +
                      std::to_string(w.rows()) + " b.size()" +
                      std::to_string(b.size()));
            }
            
            return X * w + Eigen::VectorXd::Constant(X.rows(), b(0));
            
        }

        std::map<std::string, Eigen::VectorXd> backward(const Eigen::VectorXd& z_grad, const Eigen::MatrixXd& X) const{

            if (z_grad.size() == 0 || X.rows() == 0){
                throw std::runtime_error("At least one input is empty! z_grad.size(): " +
                      std::to_string(z_grad.size()) + " X.size(): " +
                      std::to_string(X.size()));
            }
            if (X.rows() != z_grad.size()){
                throw std::runtime_error("Shape Missmatch: X.rows(): " + 
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

      double forward(const Eigen::VectorXd& y, const Eigen::VectorXd& y_hat) const {

        if (y.size() != y_hat.size()) {
          throw std::runtime_error("Shape Mismatch: y.size(): " +
            std::to_string(y.size()) + " != y_hat.size(): " +
            std::to_string(y_hat.size()));
        }
        
        // binary cross-entropy loss is:
        // -(y*log(y_hat) + (1-y)*log(1-y_hat))
        Eigen::VectorXd l = -(y.array() * y_hat.array().log() + (1.0 - y.array()) * (1.0 - y_hat.array()).log());
        
        return l.sum() / y.size();
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
        
        // gradient of the binary cross-entropy loss with respect to predictions
        // the gradient is X^T * (y_hat - y) / batch_size
        return (X.transpose() * (y_hat - y)) / y.size();
      }

};



class Params {

    public:

        Params(int num_feats = 1, double scaling_factor = 0.0, int seed = 42):
            num_feats(num_feats), scaling_factor(scaling_factor), seed(seed) {}

        std::map<std::string, Eigen::VectorXd> init_params() const {

            if (num_feats <= 0) {
                throw std::runtime_error("Number of features must be positive: " + std::to_string(num_feats));
            }

            std::map<std::string, Eigen::VectorXd> params;
            Eigen::VectorXd ws(num_feats);
            std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
            std::normal_distribution<> dist(0.0, 0.01);

            for (int i = 0; i < num_feats; ++i) {
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

    private:
        int num_feats;
        double scaling_factor;
        int seed;
};



   
    
class DataProcessor {

    public:

        DataProcessor(bool has_header = true) : header(has_header) {}

        Eigen::MatrixXd load_data(const std::string& path, int num_feats, double proportion = 1.0) {

            std::ifstream data(path);

            if (!data.is_open()) {
                throw std::runtime_error("Could not open the data!");
            }

            // count number of rows
            int row_count = 0;
            std::string line;
            while (std::getline(data, line)) {
                if (!line.empty()) {
                    row_count++;
                }
            }

            // reset file stream
            data.clear();
            data.seekg(0);

            if (header) {
                std::getline(data, line); // skip header
                row_count--;
            }

            int max_rows = static_cast<int>(row_count * proportion);
            Eigen::MatrixXd data_array(max_rows, num_feats);

            int j = 0;
            while (j < max_rows && std::getline(data, line)) {

                if (line.empty()) continue;

                std::vector<double> feats;
                std::stringstream ss(line);
                std::string val;

                while (std::getline(ss, val, ',')) {
                    feats.push_back(std::stod(val));
                }

                if (feats.size() >= static_cast<size_t>(num_feats)) {
                    for (int i = 0; i < num_feats; ++i) {
                        data_array(j, i) = feats[i];
                    }
                    j++;
                }
            }

            return data_array.topRows(j); // in case fewer rows were read than allocated
        }

        void split_data(
            const Eigen::MatrixXd& data,
            Eigen::MatrixXd& train_data,
            Eigen::MatrixXd& test_data,
            Eigen::MatrixXd& val_data,
            bool use_val = false) const {
            
            // check if total rows match
            if (!use_val) {
                if (data.rows() != train_data.rows() + test_data.rows()) {
                    throw std::runtime_error("Shape Mismatch!");
                }
            } else {
                if (data.rows() != train_data.rows() + test_data.rows() + val_data.rows()) {
                    throw std::runtime_error("Shape Mismatch!");
                }
            }
            
            // copy training data
            for (int i = 0; i < train_data.rows(); i++) {
                for (int j = 0; j < data.cols(); j++) {
                    train_data(i, j) = data(i, j);
                }
            }
            
            if (use_val) {
                // copy validation data
                for (int i = 0; i < val_data.rows(); i++) {
                    for (int j = 0; j < data.cols(); j++) {
                        val_data(i, j) = data(train_data.rows() + i, j);
                    }
                }
                
                // copy test data
                for (int i = 0; i < test_data.rows(); i++) {
                    for (int j = 0; j < data.cols(); j++) {
                        test_data(i, j) = data(train_data.rows() + val_data.rows() + i, j);
                    }
                }
            } else {
                // copy test data when no validation set
                for (int i = 0; i < test_data.rows(); i++) {
                    for (int j = 0; j < data.cols(); j++) {
                        test_data(i, j) = data(train_data.rows() + i, j);
                    }
                }
            }
        }

    private:

        bool header;

};



class Train{

    public:
        Train(
            int epochs = 1000, int val_freq = 100, int loss_freq = 100, int batch_size = 100,
            bool use_val = false, double scaling_factor = 0.2, unsigned int seed = 42): 
              epochs(epochs), val_freq(val_freq), batch_size(batch_size), loss_freq(loss_freq),
              use_val(use_val), scaling_factor(scaling_factor), seed(seed) {}

        void trainer(const Eigen::MatrixXd& train_data, const Eigen::MatrixXd& val_data, const Eigen::MatrixXd& test_data) const {

            if (batch_size > train_data.rows()) {
                throw std::runtime_error("Batch size (" + std::to_string(batch_size) +
                                         ") must be <= number of samples (" + std::to_string(train_data.rows()) + ")");
            }

            if (epochs <= 0) {
                throw std::runtime_error("Number of epochs must be positive: " + std::to_string(epochs));
            }

            DataProcessor d;
            ComputeZ ford;
            Sigmoid act;
            Params par;
            LogLoss cost;

            // TODO: complete the implementaion of Train!!!

        }


    private:

        double scaling_factor;
        unsigned int seed;
        bool use_val;
        int epochs;
        int val_freq;
        int batch_size;
        int loss_freq;


};
    








int main (){

    Eigen::MatrixXd X(2, 3);
    X << 1, 2, 3,  4, 5, 6;
    Eigen::VectorXd W(3);
    W << 1, 2, 3;
    Eigen::VectorXd b(2);
    b << 1, 2;
    //std::map<std::string, Eigen::VectorXd> p;
    //p["W"] = W;
    //p["b"] = b;


    ComputeZ lr;
    lr.forward(X, W, b);

    return 1;
}