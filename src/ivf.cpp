#include "ivf.hpp"
#include "kmeans.hpp"
#include "distance.hpp"
#include <queue>
#include <algorithm>

InvertedFileIndex::InvertedFileIndex(const collection &collection, int nlist, int nprobe):
Index(collection), nlist(nlist), nprobe(nprobe){

    uint32_t dim = coll._dim();
    std::vector<float> trainBuffer;         // all living vectors into a vector float
    std::vector<uint32_t> aliveIdx;         // translation table: buffer row → collection index
    trainBuffer.reserve(static_cast<size_t>(coll._size()) * dim);
    aliveIdx.reserve(coll._size());

    for(uint32_t i = 0 ; i < coll._size() ; ++i){
        if(coll._isdead(i)) continue;
        std::vector<float> v = coll.get_vector(i);
        trainBuffer.insert(trainBuffer.end(), v.begin(), v.end());
        aliveIdx.push_back(i);
    }

    kmeans km(nlist, 20, static_cast<int>(dim));
    km.fit_(trainBuffer);

    centroids = km.get_centroids();

    const std::vector<int> &assignments = km.get_assignments();
    InvertedList.assign(nlist, {});
    for(size_t row = 0 ; row < aliveIdx.size() ; ++row){
        int cluster = assignments[row];
        InvertedList[cluster].push_back(aliveIdx[row]);
    }
}

std::vector<int> InvertedFileIndex::findNearestCentroids(const std::vector<float> &queryVector) const{
    auto cmp = [&](const auto& a, const auto& b) { return a.first < b.first; };
    std::priority_queue<
        std::pair<float, int>,
        std::vector<std::pair<float, int>>,
        decltype(cmp)
    > topProbes(cmp);

    for(int c = 0 ; c < nlist ; ++c){
        float d = optimised_distance(queryVector, centroids[c]);

        if(topProbes.size() < static_cast<size_t>(nprobe))
            topProbes.push({d, c});
        else if(topProbes.top().first > d){
            topProbes.pop();
            topProbes.push({d, c});
        }
    }

    std::vector<int> result;
    result.reserve(topProbes.size());
    while(!topProbes.empty()){
        result.push_back(topProbes.top().second);
        topProbes.pop();
    }
    return result; 
}

std::vector<Neighbor> InvertedFileIndex::searchClusters(const std::vector<float> &queryVector, const uint32_t k, const std::vector<int> &clusterIdx) const{
    auto cmp = [](const auto& a, const auto& b) { return a.first < b.first; };
    std::priority_queue<
        std::pair<float, uint32_t>,
        std::vector<std::pair<float, uint32_t>>,
        decltype(cmp)
    > topK(cmp);

    for(int c : clusterIdx){
        for(uint32_t idx : InvertedList[c]){
            if(coll._isdead(idx)) continue; 

            float d = optimised_distance(queryVector, coll.get_vector(idx));

            if(topK.size() < k)
                topK.push({d, idx});
            else if(topK.top().first > d){
                topK.pop();
                topK.push({d, idx});
            }
        }
    }

    std::vector<Neighbor> result;
    result.reserve(topK.size());
    while(!topK.empty()){
        auto t = topK.top();
        topK.pop();
        result.push_back(Neighbor{t.first, coll._idmap(t.second)});
    }
    std::reverse(result.begin(), result.end());
    return result;
}

std::vector<Neighbor> InvertedFileIndex::search(const std::vector<float> &queryVector, const uint32_t k){
    std::vector<int> probes = findNearestCentroids(queryVector);
    return searchClusters(queryVector, k, probes);
}