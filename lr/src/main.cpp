#include <iostream>
#include <Eigen/Dense>
#include <stdexcept>
#include <random>
#include <map>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include "data_prep.hpp"
#include "utils.hpp"





// train a linear regression model

int main() {
    try {
        // load Boston Housing dataset
        Eigen::MatrixXd X;
        Eigen::VectorXd y;
        std::string filename = "../src/housing.csv"; // Adjust path to your CSV
        if (!load_boston_housing(filename, X, y)) {
            throw std::runtime_error("Failed to load dataset");
        }

        // preprocess: standardize features and center target
        standardize(X, y);

        // print dataset info
        std::cout << "Dataset loaded: " << X.rows() << " samples, " << X.cols() << " features\n";
        std::cout << "First 5 rows of X:\n" << X.topRows(5) << "\n\n";
        std::cout << "First 5 values of y:\n" << y.head(5) << "\n\n";

        // train model
        Train trainer;
        trainer.trainer(X, y, 5000, 0.1, 100);

        // final predictions
        ForwardLR ford;
        Eigen::VectorXd y_hat = ford.forward(X, trainer.params["W"], trainer.params["b"]);
        MSE mse;
        double final_mse = mse.mean_squared_error(y_hat, y);

        // print results
        std::cout << "\nFinal Weights W:\n" << trainer.params["W"] << "\n\n";
        std::cout << "Final Bias b: " << trainer.params["b"](0) << "\n\n";
        std::cout << "True y (first 5):\n" << y.head(5) << "\n\n";
        std::cout << "Predicted y_hat (first 5):\n" << y_hat.head(5) << "\n\n";
        std::cout << "Final MSE: " << final_mse << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}