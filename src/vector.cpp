#include"vector.hpp"
#include"limits"
using namespace std;

collection :: collection(uint32_t dim): dim(dim){}

bool collection::insert_vector(const string &id, const vector<float> &inputVector){
    if(maps.count(id))
        return false; //todo
    if(inputVector.size()!=dim)
        return false; // todo
    
    maps[id]=data.size();
    isdead.push_back(false);
    idmap.push_back(id);
    for(int i = 0 ; i < inputVector.size() ; ++i)
        data.push_back(inputVector[i]);
    return true;
}

bool collection :: remove_vector(const string &id){
    auto iter = maps.find(id);
    if(iter == maps.end())
        return false; //todo

    uint64_t offset = iter->second;
    uint32_t index = offset/dim;
    isdead[index]=true;
    return true;
}

vector<float> collection :: get_vector(const string &id) const{
    vector<float> returnVector(dim, 0.0);

    auto iter = maps.find(id);
    if(iter == maps.end())
        return returnVector; //todo

    uint64_t offset = iter->second;
    uint32_t index = offset/dim;

    if(isdead[index])
        return returnVector; //todo

    for(int i = 0 ; i < dim ; i++){
        returnVector[i] = data[offset + i];
    }
    return returnVector;
}

std::vector<float> collection :: get_vector(const uint64_t index) const{
    vector<float> returnVector(_dim(), 0.0);
    uint64_t offset = index * _dim();

    if(isdead[index])
        return returnVector; //todo

    for(size_t i = 0 ; i < _dim() ; i++){
        returnVector[i] = data[offset+i];
    }
    
    return returnVector;
}

inline uint64_t collection :: _offset(const std::string &id) const{
    auto iter = maps.find(id);
    if (iter == maps.end()) {
        // Return max 64bit int //todo
        return std::numeric_limits<uint64_t>::max();
    }
    
    return iter->second;
}