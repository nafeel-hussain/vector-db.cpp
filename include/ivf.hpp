#pragma once
#include "vector.hpp"
#include "search.hpp"

// ivf  search method
class InvertedFileIndex: public Index{
    std::vector<std::vector<float>> centroids;
    std::vector<std::vector<uint32_t>> InvertedList; // InvertedList[c] = collection indices assigned to cluster c
    int nlist;                                       // no of clusters
    int nprobe;                                      // how many clusters to look to find match

    //returns index of clusters
    std::vector<int> findNearestCentroids(const std::vector<float> &queryVector) const; 
    std::vector<Neighbor> searchClusters(const std::vector<float> &queryVector, const uint32_t k,const std::vector<int> &clusterIdx) const;

    public:
    InvertedFileIndex(const collection &collection, int nlist, int nprobe);
    std::vector<Neighbor> search(const std::vector<float> &queryVector,const uint32_t k) override;
    int get_nlist() const { return nlist; }
    int get_nprobe() const { return nprobe; }
    void set_nprobe(int nprobe_) { nprobe = nprobe_; }
};
