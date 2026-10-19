#include "distance.hpp"
#include "debug.h"
#include <cmath>

//Euclidean distance = sqrt(sum over(i) (|A(i) - B(i)|^2))
float Euclidean_distance(const std::vector<float> &vec1, const std::vector<float> &vec2){
    VDB_ASSERT(vec1.size() == vec2.size(), "Euclidean distance requires vectors of equal dimensions");

    float distance = 0.0;
    for(std::size_t i = 0 ; i < vec1.size() ; i++){
        distance += (vec1[i]-vec2[i])*(vec1[i]-vec2[i]);
    }

    return std::sqrt(distance);
}

// magnitude = sqrt(sum over(i) (|A(i)|^2))
float magnitude_(const std::vector<float> &vec){
    float magnitude = 0.0;
    for(std::size_t i = 0 ; i < vec.size() ; i++){
        magnitude += (vec[i]*vec[i]);
    }

    return std::sqrt(magnitude);
}

// cosing similartiy = (A dot B) / (|A|*|B|)
float cosine_similarity(const std::vector<float> &vec1, const std::vector<float> &vec2){
    VDB_ASSERT(vec1.size() == vec2.size(), "cosine similarity requires vectors of equal dimensions");

    float dotProduct = dot_product(vec1,vec2);
    float magnitudeVec1 = magnitude_(vec1);
    float magnitudeVec2 = magnitude_(vec2);

    if(magnitudeVec1 == 0 || magnitudeVec2 == 0) {
        DLOG("cosine similarity is undefined for a zero-magnitude vector");
        return NAN;
    }

    float cosineSimilarity = dotProduct / (magnitudeVec1 * magnitudeVec2);
    return cosineSimilarity;
}

// dot product = sum over(i) (|A(ij) * B(ij)|)
float dot_product(const std::vector<float> &vec1, const std::vector<float> &vec2){
    VDB_ASSERT(vec1.size() == vec2.size(), "dot product requires vectors of equal dimensions");

    float dotProduct = 0.0;
    for(std::size_t i = 0 ; i < vec1.size() ; i++){
        dotProduct += (vec1[i]*vec2[i]);
    }

    return dotProduct;
}

//optimised distance (L2) = sum over(i) (|A(i) - B(i)|^2)
float optimised_distance(const std::vector<float> &vec1, const std::vector<float> &vec2){
    VDB_ASSERT(vec1.size() == vec2.size(), "distance requires vectors of equal dimensions");

    float distance = 0.0;
    for(std::size_t i = 0 ; i < vec1.size() ; i++){
        distance += (vec1[i]-vec2[i])*(vec1[i]-vec2[i]);
    }

    return distance;
}