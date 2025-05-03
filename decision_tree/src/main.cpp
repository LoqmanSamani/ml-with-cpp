#include "utils.hpp"
#include "data_proc.hpp"

int main() {
    try {
        DataProcessor dp;
        // generate synthetic binary classification data for training
        Eigen::MatrixXd means(2, 2);
        Eigen::MatrixXd stds(2, 2);
        Eigen::VectorXi labels = Eigen::VectorXi::Zero(2000); // 2000 points (1000 per class)
        means << 1, 1,
                 3, 3;
        stds << 0.5, 0.5,
                0.5, 0.5;
        Eigen::MatrixXd train_data = dp.generate_data(means, stds, labels, 1000);

        // train the decision tree
        std::vector<int> indices(train_data.rows());
        for (int i = 0; i < train_data.rows(); ++i) {
            indices[i] = i;
        }
        DecisionTree dt(3, 0.5, 5); // max_depth=3, default_threshold=0.5, min_samples_split=5
        dt.run(train_data, labels, indices, "root", 0);

        // generate test data with a new labels vector
        Eigen::VectorXi labels_test = Eigen::VectorXi::Zero(1000); // 1000 points (500 per class)
        Eigen::MatrixXd test_data = dp.generate_data(means, stds, labels_test, 500);

        // predict on test data, passing training labels for fallback
        Eigen::VectorXi predictions = dt.predict(test_data, labels);

        // compute accuracy
        double correct = 0.0;
        for (Eigen::Index i = 0; i < predictions.size(); ++i) {
            if (predictions(i) == labels_test(i)) {
                correct++;
            }
        }
        double accuracy = (correct / predictions.size()) * 100;
        std::cout << "Prediction Accuracy: " << accuracy << "%" << std::endl;

        // plot predictions
        dp.plot_clusters(test_data, predictions, Eigen::MatrixXd(), 0, "dt_predict.png", "test", "Decision Tree Predictions");
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}