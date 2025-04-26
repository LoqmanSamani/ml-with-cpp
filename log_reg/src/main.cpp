#include <iostream>
#include <Eigen/Dense>
#include <stdexcept>
#include <fstream>
#include "data_proc.hpp"
#include "utils.hpp"




int main() {

    try {

        DataProcessor dp(true);
        Eigen::MatrixXd data = dp.load_data("../src/breast_cancer.csv", 31, 1.0);
        Eigen::MatrixXd train_data(450, 31);
        Eigen::MatrixXd test_data(69, 31);
        Eigen::MatrixXd val_data(50, 31);

        dp.split_data(data, train_data, test_data, val_data, true);

        Train t(15000, 0.007, 1000, 1000, 100, 10, true, 0.001, 0.1, true, 32);

        t.trainer(train_data, val_data, test_data, true);

        plot_losses(t.get_losses(), t.get_val_losses(), "lr_loss_plot.png", "plots/",
                    "Logistic Regression Loss", "Epoch", "Binary Cross-Entropy Loss");

        std::ofstream out("losses.csv");

        for (size_t i = 0; i < t.get_losses().size(); ++i) {
            out << t.get_losses()[i] << "," << (i % 100 == 0 ? t.get_val_losses()[i / 100] : 0) << "\n";
        }

        out.close();
        std::cout << "Losses saved to losses.csv\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}