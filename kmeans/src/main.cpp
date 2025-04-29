#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <random>
#include "utils.hpp"
#include "data_proc.hpp"





int main() {
    try {
        DataProcessor dp;
        Train t(20, 0.0001);

        Eigen::MatrixXd means(10, 2);
        Eigen::MatrixXd stds(10, 2);
        Eigen::VectorXi labels = Eigen::VectorXi::Zero(10000);
        means << 1, 1,
                 3, 3,
                 5, 5,
                 4, 4,
                 4, 8,
                 2, 6,
                 5, 1,
                 3, 1,
                 1, 8,
                 8, 1;
        stds << 0.5, 0.4, 0.7, 0.2, 0.9, 0.5, 0.6, 0.8,
                0.4, 0.6, 0.5, 0.5, 0.1, 0.7, 0.1, 0.9,
                0.7, 0.9, 0.5, 0.4;

        Eigen::MatrixXd gen_data(10000, 2);
        gen_data = dp.generate_data(means, stds, labels, 1000);

        dp.plot_clusters(gen_data, labels, means, 10, "gen_data.png", "test", "Clusters (actual clusters)");

        // standardize data before training
        dp.standardize(gen_data);
        t.trainer(gen_data, 10);

        Eigen::MatrixXd centroids = t.get_centroids();
        int num_clusters = centroids.rows();
        Eigen::VectorXi pred_labels = t.get_assign_indices();
        
        dp.plot_clusters(gen_data, pred_labels, centroids, num_clusters, "train.png", "test", "Clusters (output of k-means clustering algorithm)");


        double accuracy = dp.compute_accuracy(labels, pred_labels, num_clusters);
        std::cout << "Clustering Accuracy: " << accuracy * 100 << "%" << std::endl;
        std::cout << "Final WCSS: " << t.compute_wcss(gen_data) << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}