#ifndef DATA_PREP_HPP
#define DATA_PREP_HPP

#include <iostream>
#include <Eigen/Dense>
#include <map>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <filesystem>

namespace fs = std::filesystem;









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

        void prepare_batch(const Eigen::MatrixXd& data, Eigen::MatrixXd& X_batch, Eigen::VectorXd& y_batch, std::vector<int>& indices) const {

            for (int i = 0; i < X_batch.rows(); i++){

                for (int j = 0; j < X_batch.cols(); j++){
                    X_batch(i, j) = data(indices[i], j);   
                }

                y_batch[i] = data(indices[i], data.cols()-1);
            }
        }

    private:

        bool header;
};




void standardize(Eigen::MatrixXd& data) {

    for (Eigen::Index j = 0; j < data.cols()-1; ++j) {

        double mean = data.col(j).mean();
        double std = std::sqrt((data.col(j).array() - mean).square().mean());

        if (std > 0) {
            data.col(j) = (data.col(j).array() - mean) / std;
        } else {
            data.col(j).setZero();
        }
    }
}


// Plot training and validation losses using Gnuplot
void plot_losses(const std::vector<double>& train_losses, const std::vector<double>& val_losses,
    const std::string& filename, const std::string& plot_dir,
    const std::string& title = "Training and Validation Loss",
    const std::string& xlabel = "Epoch",
    const std::string& ylabel = "Binary Cross-Entropy Loss") {

    if (train_losses.empty()) {
    throw std::runtime_error("Train losses vector cannot be empty");
    }

    fs::create_directories(plot_dir);
    fs::path plot_dir_path = fs::path(plot_dir);
    fs::path data_path = plot_dir_path / "losses.dat";
    fs::path script_path = plot_dir_path / "plot.gp";
    fs::path output_path = plot_dir_path / filename;

    std::ofstream data_file(data_path);

    if (!data_file.is_open()) {
    throw std::runtime_error("Failed to open " + data_path.string() + " for writing");
    }

    for (size_t i = 0; i < train_losses.size(); ++i) {
    double val_loss = (i % 100 == 0 && i / 100 < val_losses.size()) ? val_losses[i / 100] : 0;
    data_file << i << " " << train_losses[i] << " " << val_loss << "\n";
    }

    data_file.close();

    std::ofstream script(script_path);

    if (!script.is_open()) {
    throw std::runtime_error("Failed to open " + script_path.string() + " for writing");
    }

    script << "set terminal png size 800,600\n";
    script << "set output '" << output_path.string() << "'\n";
    script << "set title '" << title << "'\n";
    script << "set xlabel '" << xlabel << "'\n";
    script << "set ylabel '" << ylabel << "'\n";
    script << "set grid\n";
    script << "set yrange [0:0.8]\n"; // 
    script << "plot '" << data_path.string() << "' using 1:2 with lines title 'Train Loss' lw 2, ";
    script << "'" << data_path.string() << "' using 1:3 with lines title 'Val Loss' lw 2\n";
    script.close();

    std::string command = "/usr/bin/gnuplot " + script_path.string(); // Fixed Gnuplot path
    int result = system(command.c_str());

    if (result != 0) {
    throw std::runtime_error("Gnuplot execution failed for " + script_path.string());
    }

    if (!fs::exists(output_path)) {
    throw std::runtime_error("Plot file not created: " + output_path.string());
    }
    
    std::cout << "Gnuplot saved to " << output_path.string() << std::endl;
}





#endif