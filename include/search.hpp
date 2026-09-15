#include <iostream>
#include <queue>
#include <vector>
#include "vector.hpp"

//return type for each search function 
struct Neighbor{
    float distance; // L2 norm
    std::string id;
};

//abstract class later to be inherited by searching algorithms.
class Index {
    protected:
    //ref of coll used for calculations.
    const collection &coll;                                    
    public:
    Index(const collection &collection): coll(collection) {}
    virtual ~Index(){};       
    //func to be overwritten by searching methods                                 
    virtual std::vector<Neighbor> search(const std::vector<float> &queryVector,const uint32_t k) = 0;           
};


// brute force search method
class BruteForceSearch: public Index{
    public:
    BruteForceSearch(const collection &collection):Index(collection){}
    std::vector<Neighbor> search(const std::vector<float> &queryVector,const uint32_t k);
};