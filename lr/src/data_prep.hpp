#ifndef DATA_PREP_HPP
#define DATA_PREP_HPP

#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>




bool load_boston_housing(const std::string& filename, Eigen::MatrixXd& X, Eigen::VectorXd& y) {

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    std::vector<std::vector<double>> data;
    std::string line;
    std::getline(file, line); // Skip header: RM,LSTAT,PTRATIO,MEDV

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        std::string val;

        while (std::getline(ss, val, ',')) {
            try {
                row.push_back(std::stod(val));
            } catch (...) {
                std::cerr << "Warning: Invalid value in CSV: " << val << std::endl;
                continue;
            }
        }

        if (row.size() == 4) {
            data.push_back(row);
        } else {
            std::cerr << "Warning: Skipping row with " << row.size() << " columns\n";
        }
    }

    file.close();

    if (data.empty()) {
        std::cerr << "Error: No valid data loaded\n";
        return false;
    }

    X.resize(data.size(), data[0].size() - 1);
    y.resize(data.size());

    for (size_t i = 0; i < data.size(); ++i) {

        for (size_t j = 0; j < data[0].size() - 1; ++j) {
            X(i, j) = data[i][j];
        }
        y(i) = data[i][data[0].size() - 1];
    }

    return true;
}


void standardize(Eigen::MatrixXd& X, Eigen::VectorXd& y) {

    for (Eigen::Index j = 0; j < X.cols(); ++j) {

        double mean = X.col(j).mean();
        double std = std::sqrt((X.col(j).array() - mean).square().mean());

        if (std > 0) {
            X.col(j) = (X.col(j).array() - mean) / std;
        } else {
            X.col(j).setZero();
        }
    }

    double y_mean = y.mean();
    double y_std = std::sqrt((y.array() - y_mean).square().mean());

    if (y_std > 0) {
        y = (y.array() - y_mean) / y_std;
    } else {
        y.setZero();
    }
}


void plot_losses(const std::vector<double>& losses, const std::string& filename,
                 const std::string& title = "Training Loss Over Epochs",
                 const std::string& xlabel = "Epoch",
                 const std::string& ylabel = "Mean Squared Error") {
    
    if (losses.empty()) {
        throw std::runtime_error("No losses to plot");
    }

    std::ofstream data_file("losses.dat");

    if (!data_file.is_open()) {
        throw std::runtime_error("Failed to open losses.dat for writing");
    }

    for (size_t i = 0; i < losses.size(); ++i) {
        data_file << i << " " << losses[i] << "\n"; // Epoch, Loss
    }

    data_file.close();

    std::ofstream script("plot.gp");

    if (!script.is_open()) {
        throw std::runtime_error("Failed to open plot.gp for writing");
    }

    script << "set terminal png size 800,600\n"; 
    script << "set output '" << filename << "'\n"; 
    script << "set title '" << title << "'\n"; 
    script << "set xlabel '" << xlabel << "'\n"; 
    script << "set ylabel '" << ylabel << "'\n"; 
    script << "set grid\n";
    script << "plot 'losses.dat' with lines title 'Loss' lw 2\n"; 
    script.close();

    int result = system("/gnuplot plot.gp");

    if (result != 0) {
        throw std::runtime_error("Gnuplot execution failed");
    }


    std::cout << "Gnuplot saved to " << filename << std::endl;
}



#endif // DATA_PREP_HPP