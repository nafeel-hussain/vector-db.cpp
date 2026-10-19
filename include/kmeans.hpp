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
        VDB_ASSERT(nclusters > 0, "k-means needs at least one cluster");
        VDB_ASSERT(maxIter > 0, "k-means needs at least one iteration");
        VDB_ASSERT(dim > 0, "k-means dimension must be positive");
        center.resize(nclusters, std::vector<float>(dim, 0.0f));
        DLOG("created k-means with " << nclusters << " clusters, dimension " << dim);
    }

    const std::vector<std::vector<float>>& get_centroids() const { return center; }
    const std::vector<int>& get_assignments() const { return assignments; }
    void fit_(const std::vector<float> &vecs);
    int predict_(const std::vector<float> &queryvec) const;
};