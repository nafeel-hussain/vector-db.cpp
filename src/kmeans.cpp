#include "kmeans.hpp"
#include "distance.hpp"
#include <random>
#include <limits>
#include <algorithm>
#include <cassert>

// lloyd clustering algorithm
void kmeans::fit_(const std::vector<float> &vecs) {
    int count = static_cast<int>(vecs.size() / dim);
    assert(count >= nclusters);
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

            attached[i] = index;
            clusterSize[index]++;
        }

        //recompute the centroids
         std::vector<std::vector<float>> newCenter(nclusters, std::vector<float>(dim, 0.0f));
        for(int i = 0 ; i < count ; i++){
            size_t base = static_cast<size_t>(i) * dim;
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
        attached[i] = index;
    }
    assignments = std::move(attached);
}

int kmeans::predict_ (const std::vector<float> &queryvec) const{
    float mini = std::numeric_limits<float>::max();
    int index = -1;

    for(int j = 0 ; j < nclusters ; j++){
        float d = optimised_distance(queryvec, center[j]);
        if(mini > d){
            mini = d;
            index = j;
        }
    }
    
    return index;
}