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


        void split_data(const Eigen::MatrixXd& data, Eigen::MatrixXd& train_data, Eigen::MatrixXd& test_data) const 
        {
            // check if total rows match
            if (data.rows() != train_data.rows() + test_data.rows()) 
                {
                    throw std::runtime_error("Shape Mismatch!");
                }
            
            
            // copy training data
            for (int i = 0; i < train_data.rows(); i++) 
            {
                for (int j = 0; j < data.cols(); j++) 
                {
                    train_data(i, j) = data(i, j);
                }
            }
            
            // copy test data
            for (int i = 0; i < test_data.rows(); i++) 
            {
                for (int j = 0; j < data.cols(); j++) 
                {
                    test_data(i, j) = data(train_data.rows() + i, j);
                }
            }
        }


        void standardize(Eigen::MatrixXd& data) 
        {
            for (Eigen::Index j = 0; j < data.cols()-1; ++j) 
            {
                double mean = data.col(j).mean();
                double std = std::sqrt((data.col(j).array() - mean).square().mean());
        
                if (std > 0) 
                {
                    data.col(j) = (data.col(j).array() - mean) / std;
                } 
                else 
                {
                    data.col(j).setZero();
                }
            }
        }


        Eigen::MatrixXd generate_data(Eigen::MatrixXd& means, Eigen::MatrixXd& stds, Eigen::VectorXi& labels, int num_points = 100)
        {

            if (means.rows() != stds.rows() || means.cols() != 2 || stds.cols() != 2)
            {
                throw std::runtime_error("Shape Mismatch");
            }

            int data_len = num_points * means.rows();

            if (labels.size() != data_len)
            {
                throw std::runtime_error("Shape Mismatch: labels.size() != data_len");
            }

            Eigen::MatrixXd gen_data = Eigen::MatrixXd::Zero(data_len, 2);

            std::default_random_engine generator;

            int count = 0;
            for (size_t i = 0; i < means.rows(); i++)
            {
                std::normal_distribution<double> distr1(means(i, 0), stds(i, 0));
                std::normal_distribution<double> distr2(means(i, 1), stds(i, 1));

                for (size_t j = 0; j < num_points; j++)
                {
                    double x = distr1(generator);
                    double y = distr2(generator);
                    gen_data(count + j, 0) = x;
                    gen_data(count + j, 1) = y;
                    labels(count + j) = i;
                }
                count += num_points;
            }
            
            return gen_data;

        }


        // compute clustering accuracy by finding the best permutation of predicted labels
        double compute_accuracy(const Eigen::VectorXi& true_labels, const Eigen::VectorXi& pred_labels, int num_clusters)
        {
            if (true_labels.size() != pred_labels.size()) 
            {
                throw std::runtime_error("Label sizes mismatch");
            }

            // build a confusion matrix
            Eigen::MatrixXi confusion(num_clusters, num_clusters);

            confusion.setZero();
            for (Eigen::Index i = 0; i < true_labels.size(); ++i) 
            {
                if (true_labels(i) < 0 || true_labels(i) >= num_clusters || pred_labels(i) < 0 || pred_labels(i) >= num_clusters) 
                {
                    throw std::runtime_error("Label out of bounds: true=" + std::to_string(true_labels(i)) +
                                            ", pred=" + std::to_string(pred_labels(i)));
                }
                confusion(true_labels(i), pred_labels(i))++;
            }

            // find the best permutation of predicted labels (simplified greedy approach)
            Eigen::VectorXi best_mapping(num_clusters);
            best_mapping.setConstant(-1);
            Eigen::MatrixXi conf_copy = confusion;
            int correct = 0;

            for (int iter = 0; iter < num_clusters; ++iter) 
            {
                int max_val = -1;
                int true_idx = -1;
                int pred_idx = -1;

                for (int t = 0; t < num_clusters; ++t) 
                {
                    for (int p = 0; p < num_clusters; ++p) 
                    {
                        if (conf_copy(t, p) > max_val && best_mapping(p) == -1) 
                        {
                            max_val = conf_copy(t, p);
                            true_idx = t;
                            pred_idx = p;
                        }
                    }
                }

                if (true_idx >= 0 && pred_idx >= 0)
                {
                    best_mapping(pred_idx) = true_idx;
                    correct += max_val;
                    conf_copy.row(true_idx).setZero();
                    conf_copy.col(pred_idx).setZero();
                }
            }

            return static_cast<double>(correct) / true_labels.size();
        }



        void plot_clusters(
            const Eigen::MatrixXd& data,
            const Eigen::VectorXi& labels, 
            const Eigen::MatrixXd& centroids,
            int num_clusters,
            const std::string& file_name,
            const std::string& plot_dir,
            const std::string& title = "Clusters (output of k-means clustering algorithm)",
            const std::string& xlabel = "X",
            const std::string& ylabel = "Y"
        ) 
        {

            if (data.rows() == 0 || data.cols() == 0) 
            {
                throw std::runtime_error("Data cannot be empty!");
            }
        
            // check for 2D data
            if (data.cols() != 2) 
            {
                throw std::runtime_error("Data must be 2D for scatter plot (cols = 2)");
            }
        
            
            if (labels.size() != data.rows()) 
            {
                throw std::runtime_error("Labels size must match data rows");
            }
        
            
            if (centroids.rows() != num_clusters || centroids.cols() != data.cols()) 
            {
                throw std::runtime_error("Centroids must have shape (num_clusters, data.cols)");
            }
        
            // create plot directory
            std::filesystem::create_directories(plot_dir);
            std::filesystem::path plot_dir_path = std::filesystem::path(plot_dir);
            std::filesystem::path data_path = plot_dir_path / "clusters.dat";
            std::filesystem::path centroid_path = plot_dir_path / "centroids.dat";
            std::filesystem::path script_path = plot_dir_path / "plot.gp";
            std::filesystem::path output_path = plot_dir_path / file_name;
        
            // write data points with cluster labels
            std::ofstream data_file(data_path);
            if (!data_file.is_open()) 
            {
                throw std::runtime_error("Failed to open " + data_path.string() + " for writing");
            }
            for (Eigen::Index i = 0; i < data.rows(); ++i) 
            {
                data_file << data(i, 0) << " " << data(i, 1) << " " << labels(i) << "\n";
            }
            data_file.close();
        
            // write centroids
            std::ofstream centroid_file(centroid_path);
            if (!centroid_file.is_open()) 
            {
                throw std::runtime_error("Failed to open " + centroid_path.string() + " for writing");
            }
            for (Eigen::Index i = 0; i < centroids.rows(); ++i) 
            {
                centroid_file << centroids(i, 0) << " " << centroids(i, 1) << "\n";
            }
            centroid_file.close();
        
            // write Gnuplot script
            std::ofstream script_file(script_path);
            if (!script_file.is_open()) 
            {
                throw std::runtime_error("Failed to open " + script_path.string() + " for writing");
            }
            script_file << "set terminal png size 800,600\n";
            script_file << "set output '" << output_path.string() << "'\n";
            script_file << "set title '" << title << "'\n";
            script_file << "set xlabel '" << xlabel << "'\n";
            script_file << "set ylabel '" << ylabel << "'\n";
            script_file << "set grid\n";
            // plot points with palette-based coloring by cluster index
            script_file << "set palette defined (0 'red', 1 'blue', 2 'green', 3 'purple', 4 'orange')\n";
            script_file << "plot '" << data_path.string() << "' using 1:2:3 with points pt 7 ps 1 palette title 'Clusters', ";
            // plot centroids as black stars
            script_file << "'" << centroid_path.string() << "' using 1:2 with points pt 9 ps 2 lc rgb 'black' title 'Centroids'\n";
            script_file.close();
        
            // execute gnuplot
            std::string command = "/usr/bin/gnuplot " + script_path.string();
            int result = std::system(command.c_str());
            if (result != 0) {
                throw std::runtime_error("Gnuplot execution failed for " + script_path.string());
            }
            if (!std::filesystem::exists(output_path)) {
                throw std::runtime_error("Plot file not created: " + output_path.string());
            }
            std::cout << "Gnuplot saved to " << output_path.string() << std::endl;
        }

    private:
        bool header;

};






#endif