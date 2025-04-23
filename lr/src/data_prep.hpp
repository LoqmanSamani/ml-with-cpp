#ifndef DATA_PREP_HPP
#define DATA_PREP_HPP






// updated function to load your modified Boston Housing dataset
bool load_boston_housing(const std::string& filename, Eigen::MatrixXd& X, Eigen::VectorXd& y) {

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    std::vector<std::vector<double>> data;
    std::string line;

    // skip header
    std::getline(file, line); // assumes header: RM,LSTAT,PTRATIO,MEDV

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

        if (row.size() == 4) { // 3 features + 1 target
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

    int n_samples = data.size();
    int n_features = data[0].size() - 1; // 3 features
    X.resize(n_samples, n_features);
    y.resize(n_samples);


    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            X(i, j) = data[i][j];
        }
        y(i) = data[i][n_features];
    }

    return true;
}



// function to standardize features and scale target
void standardize(Eigen::MatrixXd& X, Eigen::VectorXd& y) {

    // standardize X: (X - mean) / std
    for (int j = 0; j < X.cols(); ++j) {
        double mean = X.col(j).mean();
        double std = std::sqrt((X.col(j).array() - mean).square().sum() / X.rows());
        if (std > 0) {
            X.col(j) = (X.col(j).array() - mean) / std;
        } else {
            X.col(j).setZero(); // Avoid division by zero
        }
    }

    // scale y: (y - mean) / std (to handle large MEDV values)
    double y_mean = y.mean();
    double y_std = std::sqrt((y.array() - y_mean).square().sum() / y.size());

    if (y_std > 0) {
        y = (y.array() - y_mean) / y_std;
    } else {
        y.setZero();
    }
}



#endif