#include "kmeans.hpp"
#include "distance.hpp"
#include <random>
#include <limits>
#include <algorithm>

// lloyd clustering algorithm
void kmeans::fit_(const std::vector<float> &vecs) {
    VDB_ASSERT(dim > 0, "k-means dimension must be positive");
    VDB_ASSERT(vecs.size() % static_cast<size_t>(dim) == 0,
               "training data size must be a multiple of the k-means dimension");
    int count = static_cast<int>(vecs.size() / dim);
    VDB_ASSERT(count >= nclusters, "k-means needs at least one training vector per cluster");
    VDB_ASSERT(center.size() == static_cast<size_t>(nclusters), "k-means centroid storage is inconsistent");
    DLOG("fitting k-means to " << count << " vectors");
    std::vector<int> attached(count);           //attached[i]-> i th vector belong to which cluster index
    std::vector<int> clusterSize(nclusters,0);  //clusterSize[i] -> size of cluster i


    //assing centroids a random vecs to init
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distrib(0,count-1);
    std::vector<bool> used(count, false); // avoiding two cluster centroids to be same

    for(int i = 0 ; i < nclusters ; i++){
        int x = distrib(gen);
        while (used[x]) x = distrib(gen); 
        used[x]=true;
        center[i].assign(vecs.begin()+x*dim, vecs.begin()+(x+1)*dim);
    }

    int iter = 0;
    float error = std::numeric_limits<float>::max();
    while(iter < maxIter && error > MIN_ERROR){
        std::fill(clusterSize.begin(), clusterSize.end(), 0);

        // calc dist btw a vector and all centroids and assign it to the closest
        for(int i = 0 ; i < count ; i++){
            float mini = std::numeric_limits<float>::max();
            int index = -1;

            std::vector<float> currvec(vecs.begin()+i*dim, vecs.begin()+(i+1)*dim);

            for(int j = 0 ; j < nclusters ; j++){
                float d = optimised_distance(currvec, center[j]);
                if(mini > d){
                    mini = d;
                    index = j;
                }
            }

            VDB_ASSERT(index >= 0 && index < nclusters, "k-means failed to assign a training vector");
            attached[i] = index;
            clusterSize[index]++;
        }

        //recompute the centroids
         std::vector<std::vector<float>> newCenter(nclusters, std::vector<float>(dim, 0.0f));
        for(int i = 0 ; i < count ; i++){
            size_t base = static_cast<size_t>(i) * dim;
            VDB_ASSERT(attached[i] >= 0 && attached[i] < nclusters,
                       "k-means training assignment is out of range");
            std::vector<float> &target = newCenter[attached[i]];
            for(int s = 0 ; s < dim ; s++)
            target[s] += vecs[base + s];
        }
        for(int i = 0 ; i < nclusters ; i++){
            if(clusterSize[i] == 0) {
                // dead cluster so reseed it.
                int x = distrib(gen);
                newCenter[i].assign(vecs.begin() + static_cast<size_t>(x) * dim,
                                     vecs.begin() + static_cast<size_t>(x + 1) * dim);
                continue;
            }
            for(int s = 0 ; s < dim ; s++)
            newCenter[i][s] /= static_cast<float>(clusterSize[i]);
        }
        error = 0.0f; iter++;
        for(int i = 0 ; i < nclusters ; i++)
        error += optimised_distance(center[i], newCenter[i]);
        
        center = std::move(newCenter);
    }

    for(int i = 0 ; i < count ; i++){
        float mini = std::numeric_limits<float>::max();
        int index = -1;
        std::vector<float> currvec(vecs.begin()+i*dim, vecs.begin()+(i+1)*dim);
        for(int j = 0 ; j < nclusters ; j++){
            float d = optimised_distance(currvec, center[j]);
            if(mini > d){ mini = d; index = j; }
        }
        VDB_ASSERT(index >= 0 && index < nclusters, "k-means failed to assign a training vector");
        attached[i] = index;
    }
    assignments = std::move(attached);
    VDB_ASSERT(assignments.size() == static_cast<size_t>(count),
               "k-means assignment count does not match training vector count");
    DLOG("finished k-means after " << iter << " iterations");
}

int kmeans::predict_ (const std::vector<float> &queryvec) const{
    VDB_ASSERT(queryvec.size() == static_cast<size_t>(dim),
               "query vector dimension does not match k-means dimension");
    VDB_ASSERT(!center.empty(), "k-means has no centroids");
    float mini = std::numeric_limits<float>::max();
    int index = -1;

    for(int j = 0 ; j < nclusters ; j++){
        float d = optimised_distance(queryvec, center[j]);
        if(mini > d){
            mini = d;
            index = j;
        }
    }
    VDB_ASSERT(index >= 0 && index < nclusters, "k-means failed to select a centroid");
    return index;
}