#ifndef KROKO_MODELDATA_H_
#define KROKO_MODELDATA_H_
#include "sherpa-onnx/csrc/njson.hpp"

#include <string>
#include <vector>
#include <map>

class ModelData {
public:
    // Singleton access
    static ModelData& getInstance();

    // Deleted to prevent copying or assignment
    ModelData(const ModelData&) = delete;
    void operator=(const ModelData&) = delete;

    // Public interface
    bool loadHeader(const std::string& filepath);
    bool decryptPayload(const std::string& password);
    bool loadPayload();
    std::string getHeaderValue(const std::string& key) const;

    std::vector<uint8_t> encoder, decoder, joiner;
    std::string tokens;

private:
    // Private constructor for singleton
    ModelData() = default;

    std::map<std::string, nlohmann::json> header;
    std::vector<uint8_t> blob;
    std::string filePath;
};

#endif