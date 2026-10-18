#pragma once
#include "vector.hpp"

#define MIN_ERROR 10e-4

class kmeans{
    private:
    int nclusters;                              //num of cluster
    std::vector<std::vector<float>> center;     // centroid
    std::vector<int> assignments;               // assignments[i]=cluster index for the ith vector passed to last fit_() call
    int maxIter;
    int dim;

    public:
    kmeans(int nclusters = 50, int maxIter = 20, int dim = 128):
    nclusters(nclusters), maxIter(maxIter), dim(dim) {
        center.resize(nclusters, std::vector<float>(dim, 0.0f));
    }

    const std::vector<std::vector<float>>& get_centroids() const { return center; }
    const std::vector<int>& get_assignments() const { return assignments; }
    void fit_(const std::vector<float> &vecs);
    int predict_(const std::vector<float> &queryvec) const;
};