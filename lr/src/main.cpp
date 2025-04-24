#include <iostream>
#include <Eigen/Dense>
#include <stdexcept>
#include <fstream>
#include "data_prep.hpp"
#include "utils.hpp"



int main() {

    try {
        // load Boston Housing dataset
        Eigen::MatrixXd X;
        Eigen::VectorXd y;
        std::string filename = "../src/housing.csv";
        if (!load_boston_housing(filename, X, y)) {
            throw std::runtime_error("Failed to load dataset");
        }

        // preprocess: standardize features and center target
        standardize(X, y);

        // print dataset info
        std::cout << "Dataset loaded: " << X.rows() << " samples, " << X.cols() << " features\n";
        std::cout << "First 5 rows of X:\n" << X.topRows(5) << "\n\n";
        std::cout << "First 5 values of y:\n" << y.head(5) << "\n\n";

        // train model with mini-batches
        Train trainer(0.2, 42);
        trainer.trainer(X, y, 1500, 50, 0.003, 100);

        // visualize training losses using Gnuplot
        plot_losses(trainer.losses, "lr_loss_plot.png", "Training Loss Over Epochs (Linear Regression)",
                    "Epoch", "Mean Squared Error");

        // save losses to CSV as fallback
        std::ofstream out("losses.csv");
        for (double loss : trainer.losses) out << loss << "\n";
        out.close();
        std::cout << "Losses saved to losses.csv\n";

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