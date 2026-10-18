#pragma once
#include <cstdint>
#include<iostream>
#include<vector>
#include<unordered_map>

class collection{
    std :: vector<float> data;
    std :: unordered_map<std::string, uint64_t> maps;   // maps (id) -> offset
    std :: vector<std::string> idmap;                   // maps (index) -> id
    std :: vector<bool> isdead;
    uint32_t dim;

    public:
    collection(uint32_t dim);
    bool insert_vector(const std::string &id, const std::vector<float> &vector);
    bool remove_vector(const std::string &id);
    std::vector<float> get_vector(const std::string &id) const;
    std::vector<float> get_vector(const uint64_t index) const;

    //Helper functions to access private members
    inline bool _isdead (const uint64_t index) const { return isdead[index]; };   // return true if data[index] is dead
    inline uint32_t _dim() const { return dim; };                                 // return dimention
    inline uint32_t _size() const { return data.size()/_dim(); };                 // returns number of vectors in collection
    inline uint64_t _offset(const std::string &id) const;                         // return offset to an index
    inline std::string _idmap(const uint64_t index) const {return idmap[index]; } // returns id from index.
    inline double _memory() const { return static_cast<double> (data.size() * sizeof(float));}    // bytes used 
};