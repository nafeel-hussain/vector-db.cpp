#include"vector.hpp"
#include"limits"
using namespace std;

collection :: collection(uint32_t dim): dim(dim){
    VDB_ASSERT(dim > 0, "collection dimension must be positive");
    DLOG("created collection with dimension " << dim);
}

bool collection::insert_vector(const string &id, const vector<float> &inputVector){
    if(maps.count(id)) {
        DLOG("rejected duplicate vector id '" << id << "'");
        return false; //todo
    }
    if(inputVector.size()!=dim) {
        DLOG("rejected vector id '" << id << "' with dimension " << inputVector.size()
                  << "; expected " << dim);
        return false; // todo
    }
    
    maps[id]=data.size();
    isdead.push_back(false);
    idmap.push_back(id);
    for(size_t i = 0 ; i < inputVector.size() ; ++i)
        data.push_back(inputVector[i]);
    VDB_ASSERT(data.size() == isdead.size() * dim, "collection storage is inconsistent after insert");
    DLOG("inserted vector id '" << id << "' at index " << (isdead.size() - 1));
    return true;
}

bool collection :: remove_vector(const string &id){
    auto iter = maps.find(id);
    if(iter == maps.end()) {
        DLOG("could not remove unknown vector id '" << id << "'");
        return false; //todo
    }

    uint64_t offset = iter->second;
    VDB_ASSERT(dim > 0 && offset % dim == 0, "stored vector offset is invalid");
    uint32_t index = offset/dim;
    VDB_ASSERT(index < isdead.size(), "stored vector index is out of range");
    if(isdead[index])
        DLOG("vector id '" << id << "' was already removed");
    isdead[index]=true;
    DLOG("removed vector id '" << id << "' at index " << index);
    return true;
}

vector<float> collection :: get_vector(const string &id) const{
    vector<float> returnVector(dim, 0.0);

    auto iter = maps.find(id);
    if(iter == maps.end()) {
        DLOG("lookup for unknown vector id '" << id << "'");
        return returnVector; //todo
    }

    uint64_t offset = iter->second;
    VDB_ASSERT(dim > 0 && offset % dim == 0, "stored vector offset is invalid");
    uint32_t index = offset/dim;
    VDB_ASSERT(index < isdead.size(), "stored vector index is out of range");

    if(isdead[index])
        return returnVector; //todo

    for(size_t i = 0 ; i < dim ; i++){
        returnVector[i] = data[offset + i];
    }
    return returnVector;
}

std::vector<float> collection :: get_vector(const uint64_t index) const{
    VDB_ASSERT(index < _size(), "collection index is out of range");
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
        DLOG("offset requested for unknown vector id '" << id << "'");
        return std::numeric_limits<uint64_t>::max();
    }
    VDB_ASSERT(dim > 0 && iter->second % dim == 0, "stored vector offset is invalid");
    
    return iter->second;
}