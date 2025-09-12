// sherpa-onnx/csrc/online-recognizer.cc
//
// Copyright (c)  2023  Xiaomi Corporation
// Copyright (c)  2023  Pingfeng Luo

#include "sherpa-onnx/csrc/online-recognizer.h"

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/online-recognizer-impl.h"
#include "sherpa-onnx/csrc/text-utils.h"

#include <thread>
#include <map>
#include <functional>
#include <bits/stdc++.h>
#include<iostream>
#include<algorithm>

namespace sherpa_onnx {

  std::string escape_json_online(const std::string &s) {
    std::ostringstream o;
    for (auto c = s.cbegin(); c != s.cend(); c++) {
        switch (*c) {
        case '"': o << "\\\""; break;
        case '\\': o << "\\\\"; break;
        case '\b': o << "\\b"; break;
        case '\f': o << "\\f"; break;
        case '\n': o << "\\n"; break;
        case '\r': o << "\\r"; break;
        case '\t': o << "\\t"; break;
        default:
            if ('\x00' <= *c && *c <= '\x1f') {
                o << "\\u"
                  << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(*c);
            } else {
                o << *c;
            }
        }
    }
    return o.str();
}

std::string OnlineRecognizerResult::AsJsonString() const {
  std::string mergedText = text;

  std::vector<std::string> words_text;
  std::vector<float> words_timestamps;
  try {
  if(tokens.size() > 0) {
      std::string word = tokens[0];
      float word_timestamp = timestamps[0];
      if(tokens.size() == 1) {
        words_text.push_back(word);
        words_timestamps.push_back(word_timestamp);
      }
      for(int i = 1; i < tokens.size(); i++) {
        if(tokens[i].find_first_of(" ") == 0) {
          if(word.find_first_of(" ") == 0) {
            word = word.erase(0, 1);
          }
          words_text.push_back(word);
          words_timestamps.push_back(word_timestamp);
          word = tokens[i];
          word_timestamp = timestamps[i];
        }
        else {
          word += tokens[i];
        }
        if(i == tokens.size() - 1) {
          if(word.find_first_of(" ") == 0) {
            word = word.erase(0, 1);
          }
          words_text.push_back(word);
          words_timestamps.push_back(word_timestamp);
        }
      }
  }
  } catch (const std::exception& e) {}

  std::ostringstream os;
  os << "{";
  os << "\"text\""
     << ": ";
  os << "\"" << escape_json_online(mergedText) << "\""
     << ", ";
  if(is_final) {
    os << "\"type\": \"final\", ";
  }
  else {
    os << "\"type\": \"partial\", ";
  }
  os << "\"startedAt\": " << start_time << ", ";
  os << "\"segment\": " << segment << ", ";

  os << "\"elements\": {";

  os << "\"segments\": [";
  os << "{";
  os << "\"text\""
   << ": ";
  os << "\"" << escape_json_online(mergedText) << "\"" << ", ";
  os << "\"type\": \"segment\", ";
  os << "\"startedAt\": " << start_time << ", ";
  os << "\"segment\": " << segment;
  os << "}";
  os << "], "; //segments

  std::string sep = "";
  double prev_timestamp = 0;
  os << "\"words\": [";
  for(int i = 0; i < words_text.size(); i ++) {
    double timestamp = words_timestamps[i];
    if(timestamp <= prev_timestamp) {
      timestamp = prev_timestamp + 0.01;
    }
    prev_timestamp = timestamp;
    os << sep;
    os << "{";
    os << "\"text\""
     << ": ";
    os << "\"" << escape_json_online(words_text[i]) << "\""
      << ", ";
    os << "\"type\": \"word\", ";
    os << "\"startedAt\": " << timestamp + start_time << ", ";
    os << "\"segment\": " << segment;
    os << "}";
    sep = ", ";
  }
  os << "]"; //words

  os << "}"; //elements

  os << "}";
  return os.str();
}

void OnlineRecognizerConfig::Register(ParseOptions *po) {
  feat_config.Register(po);
  model_config.Register(po);
  endpoint_config.Register(po);
  lm_config.Register(po);
  ctc_fst_decoder_config.Register(po);


  po->Register("enable-endpoint", &enable_endpoint,
               "True to enable endpoint detection. False to disable it.");
  po->Register("max-active-paths", &max_active_paths,
               "beam size used in modified beam search.");
  po->Register("blank-penalty", &blank_penalty,
               "The penalty applied on blank symbol during decoding. "
               "Note: It is a positive value. "
               "Increasing value will lead to lower deletion at the cost"
               "of higher insertions. "
               "Currently only applicable for transducer models.");
  po->Register("hotwords-score", &hotwords_score,
               "The bonus score for each token in context word/phrase. "
               "Used only when decoding_method is modified_beam_search");
  po->Register(
      "hotwords-file", &hotwords_file,
      "The file containing hotwords, one words/phrases per line, For example: "
      "HELLO WORLD"
      "你好世界");
  po->Register("decoding-method", &decoding_method,
               "decoding method,"
               "now support greedy_search and modified_beam_search.");
  po->Register("temperature-scale", &temperature_scale,
               "Temperature scale for confidence computation in decoding.");
  po->Register(
      "rule-fsts", &rule_fsts,
      "If not empty, it specifies fsts for inverse text normalization. "
      "If there are multiple fsts, they are separated by a comma.");

  po->Register(
      "rule-fars", &rule_fars,
      "If not empty, it specifies fst archives for inverse text normalization. "
      "If there are multiple archives, they are separated by a comma.");

  po->Register("reset-encoder", &reset_encoder,
               "True to reset encoder_state on an endpoint after empty segment."
               "Done in `Reset()` method, after an endpoint was detected.");
}

bool OnlineRecognizerConfig::Validate() const {
  if (decoding_method == "modified_beam_search" && !lm_config.model.empty()) {
    if (max_active_paths <= 0) {
      SHERPA_ONNX_LOGE("max_active_paths is less than 0! Given: %d",
                       max_active_paths);
      return false;
    }

    if (!lm_config.Validate()) {
      return false;
    }
  }

  if (!hotwords_file.empty() && decoding_method != "modified_beam_search") {
    SHERPA_ONNX_LOGE(
        "Please use --decoding-method=modified_beam_search if you"
        " provide --hotwords-file. Given --decoding-method=%s",
        decoding_method.c_str());
    return false;
  }

  if (!ctc_fst_decoder_config.graph.empty() &&
      !ctc_fst_decoder_config.Validate()) {
    SHERPA_ONNX_LOGE("Errors in ctc_fst_decoder_config");
    return false;
  }

  if (!hotwords_file.empty() && !FileExists(hotwords_file)) {
    SHERPA_ONNX_LOGE("--hotwords-file: '%s' does not exist",
                     hotwords_file.c_str());
    return false;
  }

  if (!rule_fsts.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(rule_fsts, ",", false, &files);
    for (const auto &f : files) {
      if (!FileExists(f)) {
        SHERPA_ONNX_LOGE("Rule fst '%s' does not exist. ", f.c_str());
        return false;
      }
    }
  }

  if (!rule_fars.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(rule_fars, ",", false, &files);
    for (const auto &f : files) {
      if (!FileExists(f)) {
        SHERPA_ONNX_LOGE("Rule far '%s' does not exist. ", f.c_str());
        return false;
      }
    }
  }

  if (!hr.dict_dir.empty() && !hr.lexicon.empty() && !hr.rule_fsts.empty() &&
      !hr.Validate()) {
    return false;
  }

  return model_config.Validate();
}

std::string OnlineRecognizerConfig::ToString() const {
  std::ostringstream os;

  os << "OnlineRecognizerConfig(";
  os << "feat_config=" << feat_config.ToString() << ", ";
  os << "model_config=" << model_config.ToString() << ", ";
  os << "lm_config=" << lm_config.ToString() << ", ";
  os << "endpoint_config=" << endpoint_config.ToString() << ", ";
  os << "ctc_fst_decoder_config=" << ctc_fst_decoder_config.ToString() << ", ";
  os << "enable_endpoint=" << (enable_endpoint ? "True" : "False") << ", ";
  os << "max_active_paths=" << max_active_paths << ", ";
  os << "hotwords_score=" << hotwords_score << ", ";
  os << "hotwords_file=\"" << hotwords_file << "\", ";
  os << "decoding_method=\"" << decoding_method << "\", ";
  os << "blank_penalty=" << blank_penalty << ", ";
  os << "temperature_scale=" << temperature_scale << ", ";
  os << "rule_fsts=\"" << rule_fsts << "\", ";
  os << "rule_fars=\"" << rule_fars << "\", ";
  os << "reset_encoder=" << (reset_encoder ? "True" : "False") << ", ";
  os << "hr=" << hr.ToString() << ")";

  return os.str();
}

OnlineRecognizer::OnlineRecognizer(const OnlineRecognizerConfig &config)
    : impl_(OnlineRecognizerImpl::Create(config)) {}

template <typename Manager>
OnlineRecognizer::OnlineRecognizer(Manager *mgr,
                                   const OnlineRecognizerConfig &config)
    : impl_(OnlineRecognizerImpl::Create(mgr, config)) {}

OnlineRecognizer::~OnlineRecognizer() = default;

std::unique_ptr<OnlineStream> OnlineRecognizer::CreateStream() const {
  return impl_->CreateStream();
}

std::unique_ptr<OnlineStream> OnlineRecognizer::CreateStream(
    const std::string &hotwords) const {
  return impl_->CreateStream(hotwords);
}

bool OnlineRecognizer::IsReady(OnlineStream *s) const {
  return impl_->IsReady(s);
}

void OnlineRecognizer::WarmpUpRecognizer(int32_t warmup, int32_t mbs) const {
  if (warmup > 0) {
    impl_->WarmpUpRecognizer(warmup, mbs);
  }
}

void OnlineRecognizer::DecodeStreams(OnlineStream **ss, int32_t n) const {
  impl_->DecodeStreams(ss, n);
}

OnlineRecognizerResult OnlineRecognizer::GetResult(OnlineStream *s) const {
  return impl_->GetResult(s);
}

bool OnlineRecognizer::IsEndpoint(OnlineStream *s) const {
  return impl_->IsEndpoint(s);
}

void OnlineRecognizer::Reset(OnlineStream *s) const { impl_->Reset(s); }

#if __ANDROID_API__ >= 9
template OnlineRecognizer::OnlineRecognizer(
    AAssetManager *mgr, const OnlineRecognizerConfig &config);
#endif

#if __OHOS__
template OnlineRecognizer::OnlineRecognizer(
    NativeResourceManager *mgr, const OnlineRecognizerConfig &config);
#endif

}  // namespace sherpa_onnx
