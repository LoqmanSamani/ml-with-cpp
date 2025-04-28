#ifndef UTILS.HPP
#define UTILS.HPP

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <cmath> 




class KMeansInit {

    public:

        KMeansInit(int num_k, int start, int end) : num_k(num_k), start(start), end(end)
        {
            cent_indices = Eigen::VectorXi::Zero(num_k);
        }
        
        // initialize centroids with K-means++ initialization method

        Eigen::MatrixXd centroids_(const Eigen::MatrixXd& data)
        {
            if (num_k >= data.rows())
            {
                throw std::runtime_error("Number of centroids must be less than data points!");
            }

            Eigen::MatrixXd centroids = Eigen::MatrixXd::Zero(num_k, data.cols());
            
            std::random_device rand_dev;
            std::mt19937 generator(rand_dev());
            std::uniform_real_distribution<double> dis(0.0, 1.0);
            
            // randomly choose first centroid
            std::uniform_int_distribution<int> first_cent_distr(start, end);
            int first_cent_index = first_cent_distr(generator);
            centroids.row(0) = data.row(first_cent_index);
            cent_indices[0] = first_cent_index;
            
            // choose the rest of centroids
            int count = 1;
            while (count < num_k)
            {

                // calculate squared distances from each point to its nearest existing centroid
                Eigen::VectorXd dist_from_nearest(data.rows());
                
                for (int point = 0; point < data.rows(); point++)
                {
                    double min_dist = std::numeric_limits<double>::max();
        
                    for (int i = 0; i < count; i++)
                    {
                        // calculate squared Euclidean distance
                        double dist = (data.row(point) - centroids.row(i)).squaredNorm();
                        if (dist < min_dist)
                        {
                            min_dist = dist;
                        }
                    }
                    
                    dist_from_nearest(point) = min_dist; 
                }

                if (dist_from_nearest.sum() < 1e-10)
                {
                    throw std::runtime_error("Degenerate case: all points identical");
                }

                // select next centroid using weighted probability
                Eigen::VectorXd point_probs = dist_from_nearest.array() / dist_from_nearest.sum();
                double random_value = dis(generator);
                double cumulative = 0.0;
                int chosen_index = -1;
                
                for (int i = 0; i < point_probs.size(); ++i)
                {
                    cumulative += point_probs(i);
                    if (random_value <= cumulative) 
                    {
                        chosen_index = i;
                        break;
                    }
                }
                
                bool already_selected = false;
                for (int i = 0; i < count; i++) 
                {
                    if (cent_indices[i] == chosen_index) 
                    {
                        already_selected = true;
                        break;
                    }
                }
                
                if (!already_selected) 
                {
                    cent_indices[count] = chosen_index;
                    centroids.row(count) = data.row(chosen_index);
                    count++;
                }
            }
            
            return centroids;
        }
    
    private:

        int num_k;
        int start;
        int end;
        Eigen::VectorXi cent_indices; 
};




class AssignClusters {

    public:
        
        void assign_clusters(const Eigen::MatrixXd& data, const Eigen::MatrixXd& centroids, Eigen::VectorXd& assign_indices) 
        {
            if (data.rows() != assign_indices.size() || centroids.rows() >= data.rows()) 
            {
                throw std::runtime_error("Number of centroids must be less than data points!");
            }
            
            for (int point = 0; point < data.rows(); point++) 
            {
                double min_dist = std::numeric_limits<double>::max();
                int closest_cent_idx = -1;
                
                for (int i = 0; i < centroids.rows(); i++) 
                {
                    double distance = (data.row(point) - centroids.row(i)).squaredNorm();
                    
                    if (distance < min_dist) 
                    {
                        min_dist = distance;
                        closest_cent_idx = i;  
                    }
                }
                
                assign_indices(point) = closest_cent_idx;
            }
        }
    };





class UpdateCentroids {

    public:

        void update_centroids(const Eigen::MatrixXd& data, Eigen::MatrixXd& centroids, const Eigen::VectorXi& assign_indices)
        {
            if (data.rows() != assign_indices.size() || centroids.rows() >= data.rows()) {
                throw std::runtime_error("Number of centroids must be less than data points!");
            }

            for (int cent = 0; cent < centroids.rows(); cent++)
            {
                
                Eigen::VectorXd sum = Eigen::VectorXd::Zero(centroids.cols());
                int num_assignments = 0;

                for (int point = 0; point < data.rows(); point++)
                {
                    if (assign_indices(point) == cent) 
                    {
                        sum += data.row(point);
                        num_assignments++;
                    }
                }
                
                if (num_assignments == 0) {
                    throw std::runtime_error("Empty cluster for centroid " + std::to_string(cent));
                }

                centroids.row(cent) = sum / num_assignments;
            }
        }
};





class Train {

    public:

        Train(int max_epoch = 20, double threshold = 0.01) : max_epoch(max_epoch), threshold(threshold) {}

        void trainer(Eigen::MatrixXd data, int num_centroids)
        {
            if (num_centroids >= data.rows()) 
            {
                throw std::runtime_error("Number of centroids must be less than data points!");
            }

            KMeansInit init(num_centroids, 0, data.rows() - 1);
            AssignClusters assign;
            UpdateCentroids update;

            // initialize centroids
            centroids = init.centroids_(data);
            // Fix 12: Changed assign_indices to VectorXi
            assign_indices = Eigen::VectorXd::Zero(data.rows());

            for (int i = 0; i < max_epoch; i++)
            {
                Eigen::MatrixXd pre_centroids = centroids;

                assign.assign_clusters(data, centroids, assign_indices);
                update.update_centroids(data, centroids, assign_indices);

                double change = (centroids - pre_centroids).norm();
    
                if (change < threshold) 
                {
                    std::cout << "Converged after " << i + 1 << " epochs" << std::endl;
                    return;
                }
            }
            std::cout << "Reached maximum epochs (" << max_epoch << ") without convergence" << std::endl;
        }

        Eigen::MatrixXd get_centroids() 
        {
            return centroids;
        }
        
        
        Eigen::VectorXd get_assign_indices() 
        {
            return assign_indices;
        }

    private:

        int max_epoch;
        double threshold;
        Eigen::MatrixXd centroids;
        Eigen::VectorXd assign_indices;

};





class Predict
{
    public:

        Predict(Eigen::MatrixXd centroids): centroids(centroids) {}

        void assign_new_points(const Eigen::MatrixXd& new_points)
        {
            if (new_points.cols() != centroids.cols())
            {
                throw std::runtime_error("Shape Mismatch!");
            }

            assign_indices = Eigen::VectorXd::Zero(new_points.rows());
            AssignClusters assign;
            assign.assign_clusters(new_points, centroids, assign_indices);
        }

        Eigen::VectorXd get_assign_indices()
        {
            return assign_indices;
        }

    private:

        Eigen::MatrixXd centroids;
        Eigen::VectorXd assign_indices;
};





#endif 