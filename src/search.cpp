#include "search.hpp"
#include "distance.hpp"

std::vector<Neighbor> BruteForceSearch::search(const std::vector<float> &queryVector, const uint32_t k){
    // This search calculates distance between queryVector and each vector in collection
    // Then ranks them for shortest distance and returns k closest vectors to it.
    // distance(v[0])<distance(v[1])<...

    auto cmp = [](const auto& a, const auto& b) { return a.first < b.first; };
    std::priority_queue<
        std::pair<float, uint32_t>, 
        std::vector<std::pair<float, uint32_t>>, 
        decltype(cmp)
    > topK(cmp);

    for(uint32_t i = 0 ; i < coll._size(); ++i){
        if(coll._isdead(i)) continue;

        float distance = optimised_distance(queryVector, coll.get_vector(i));

        if(topK.size() < k)
        topK.push({distance,i});

        else if (topK.top().first > distance){
            topK.pop();
            topK.push({distance,i});
        }
    }

    size_t elements = topK.size();
    std::vector<Neighbor> returnVector;
    returnVector.reserve(elements);

    while(!topK.empty()){
        std::pair<float,uint32_t> t = topK.top();
        auto dist = t.first; 
        auto id = coll._idmap(t.second);

        topK.pop();
        returnVector.push_back(Neighbor{dist, id});
    }

    std::reverse(returnVector.begin(), returnVector.end()); 
    return returnVector;
}