#include "sherpa-onnx/csrc/ModelData.h"
#include <fstream>
#include <iostream>
#include <openssl/aes.h>
#include <openssl/evp.h>
#include <openssl/rand.h>

ModelData& ModelData::getInstance() {
    static ModelData instance;
    return instance;
}

bool ModelData::loadHeader(const std::string& filepath) {
    filePath = filepath;
    std::ifstream in(filepath, std::ios::binary);
    if (!in) return false;

    uint32_t headerLen;
    in.read(reinterpret_cast<char*>(&headerLen), sizeof(headerLen));
    std::vector<char> headerBuf(headerLen);
    in.read(headerBuf.data(), headerLen);

    try {
        auto j = nlohmann::json::parse(headerBuf);
        for (auto& el : j.items())
            header[el.key()] = el.value();
    } catch (...) {
        return false;
    }

    blob.assign(std::istreambuf_iterator<char>(in), {});
    return true;
}

bool ModelData::decryptPayload(const std::string& password) {
    if (blob.size() < 16) return false;
    std::vector<uint8_t> key(32, '0');
    std::copy(password.begin(), password.end(), key.begin());

    const unsigned char* iv = blob.data();
    const unsigned char* ciphertext = blob.data() + 16;
    int ciphertext_len = blob.size() - 16;

    std::vector<uint8_t> plaintext(ciphertext_len + AES_BLOCK_SIZE);

    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    int len, plaintext_len;

    EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key.data(), iv);
    EVP_DecryptUpdate(ctx, plaintext.data(), &len, ciphertext, ciphertext_len);
    plaintext_len = len;

    EVP_DecryptFinal_ex(ctx, plaintext.data() + len, &len);
    plaintext_len += len;
    EVP_CIPHER_CTX_free(ctx);
    plaintext.resize(plaintext_len);

    auto it = plaintext.begin();
    auto readBlock = [&](std::vector<uint8_t>::iterator& it) {
        uint32_t len = *reinterpret_cast<uint32_t*>(&*it);
        it += 4;
        std::vector<uint8_t> buf(it, it + len);
        it += len;
        return buf;
    };

    try {
        encoder = readBlock(it);
        decoder = readBlock(it);
        joiner  = readBlock(it);
        auto tokensData = readBlock(it);
        tokens.assign(tokensData.begin(), tokensData.end());
    } catch (const std::exception& e) {
        std::cerr << "Payload parsing error: " << e.what() << "\n";
        return false;
    }

    return true;
}

bool ModelData::loadPayload() {
    if (blob.size() < 4) return false;

    auto it = blob.cbegin();

    // Step 2: Parse the blocks
    auto readBlock = [&](std::vector<uint8_t>::const_iterator& it) {
        if (std::distance(it, blob.cend()) < 4)
            throw std::runtime_error("Invalid block header");

        uint32_t len = *reinterpret_cast<const uint32_t*>(&*it);
        it += 4;

        if (std::distance(it, blob.cend()) < len)
            throw std::runtime_error("Block size mismatch");

        std::vector<uint8_t> buf(it, it + len);
        it += len;
        return buf;
    };

    try {
        encoder = readBlock(it);
        decoder = readBlock(it);
        joiner  = readBlock(it);
        auto tokensData = readBlock(it);
        tokens.assign(tokensData.begin(), tokensData.end());
    } catch (const std::exception& e) {
        std::cerr << "Payload parsing error: " << e.what() << "\n";
        return false;
    }

    return true;
}

std::string ModelData::getHeaderValue(const std::string& key) const {
    auto it = header.find(key);
    if (it == header.end()) return "";

    // Convert JSON value to string safely
    if (it->second.is_string()) {
        return it->second.get<std::string>();
    } else {
        return it->second.dump();  // serialize any other type (bool, object, number)
    }
}
