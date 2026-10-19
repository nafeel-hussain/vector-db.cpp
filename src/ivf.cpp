#include "ivf.hpp"
#include "kmeans.hpp"
#include "distance.hpp"
#include <queue>
#include <algorithm>

InvertedFileIndex::InvertedFileIndex(const collection &collection, int nlist, int nprobe):
Index(collection), nlist(nlist), nprobe(nprobe){

    VDB_ASSERT(nlist > 0, "IVF needs at least one list");
    VDB_ASSERT(nprobe > 0 && nprobe <= nlist, "IVF nprobe must be in [1, nlist]");
    VDB_ASSERT(coll._dim() > 0, "IVF collection dimension must be positive");
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

    VDB_ASSERT(aliveIdx.size() >= static_cast<size_t>(nlist),
               "IVF needs at least nlist live vectors for training");
    DLOG("building IVF index with " << aliveIdx.size() << " live vectors, "
              << nlist << " lists and " << nprobe << " probes");
    kmeans km(nlist, 20, static_cast<int>(dim));
    km.fit_(trainBuffer);

    centroids = km.get_centroids();

    const std::vector<int> &assignments = km.get_assignments();
    VDB_ASSERT(assignments.size() == aliveIdx.size(),
               "IVF training assignments do not match live vector count");
    InvertedList.assign(nlist, {});
    for(size_t row = 0 ; row < aliveIdx.size() ; ++row){
        int cluster = assignments[row];
        VDB_ASSERT(cluster >= 0 && cluster < nlist, "k-means returned an invalid cluster assignment");
        InvertedList[cluster].push_back(aliveIdx[row]);
    }
    VDB_ASSERT(InvertedList.size() == static_cast<size_t>(nlist), "IVF list count is inconsistent");
    DLOG("finished building IVF index");
}

std::vector<int> InvertedFileIndex::findNearestCentroids(const std::vector<float> &queryVector) const{
    VDB_ASSERT(queryVector.size() == coll._dim(), "query dimension does not match IVF collection");
    VDB_ASSERT(centroids.size() == static_cast<size_t>(nlist), "IVF centroid count is inconsistent");
    VDB_ASSERT(nprobe > 0 && nprobe <= nlist, "IVF nprobe must be in [1, nlist]");
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
    VDB_ASSERT(queryVector.size() == coll._dim(), "query dimension does not match IVF collection");
    auto cmp = [](const auto& a, const auto& b) { return a.first < b.first; };
    std::priority_queue<
        std::pair<float, uint32_t>,
        std::vector<std::pair<float, uint32_t>>,
        decltype(cmp)
    > topK(cmp);

    for(int c : clusterIdx){
        VDB_ASSERT(c >= 0 && c < nlist, "IVF search received an invalid cluster index");
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
    DLOG("IVF search requested with k=" << k);
    std::vector<int> probes = findNearestCentroids(queryVector);
    std::vector<Neighbor> results = searchClusters(queryVector, k, probes);
    DLOG("IVF search inspected " << probes.size() << " lists and returned " << results.size()
              << " neighbors");
    return results;
}