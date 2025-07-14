#pragma once

#include <memory>
#include <iostream>
#include "cutlass/numeric_types.h"
#include "src/fastertransformer/utils/ini.h"

namespace fastertransformer {

enum class FetchType {
    GPU_ONLY,
    FETCH_ON_DEMAND,
    PREFETCH
};

enum class QuantType {
    NO_QUANT,
    WEIGHT_ONLY,
    SMOOTH_QUANT
};

class GlobalConfig {
public:
    using quant_t = cutlass::fp4_t;

    using weight_t = float;

    using act_t = float;

    static GlobalConfig& instance()
    {
        static GlobalConfig instance;
        return instance;
    }

    void setDefault()
    {
        loadDefault();
    }

    void loadDefault()
    {
        mINI::INIFile file("/workspace/FasterTransformer/cpp_config.ini");
        mINI::INIStructure ini;
        file.read(ini);

        arena_size = std::stoul(ini["default"]["arena_size"]);

        encoder_fetcher_mode = static_cast<FetchType>(std::stoi(ini["default"]["encoder_fetcher_mode"]));
        decoder_fetcher_mode = static_cast<FetchType>(std::stoi(ini["default"]["decoder_fetcher_mode"]));

        profiling = std::stoi(ini["default"]["profiling"]);
        detailed_timing = std::stoi(ini["default"]["detailed_timing"]);

        offload_path = ini["default"]["offload_path"];
        // disk_offload = std::stoi(ini["default"]["disk_offload"]);
        disk_offload = 0;
        
        load_from_cpp = std::stoi(ini["default"]["load_from_cpp"]);

        // use_cache = std::stoi(ini["default"]["use_cache"]);
        use_cache = 0;

        quant_mode = static_cast<QuantType>(std::stoi(ini["default"]["quant_mode"]));

        vocab_size = std::stoll(ini["default"]["vocab_size"]);

        // fetch_all = std::stoi(ini["default"]["fetch_all"]);
        fetch_all = 0;
        // forced_num_experts = std::stoi(ini["default"]["forced_num_experts"]);
        forced_num_experts = 0;

        cache_policy = ini["default"]["cache_policy"];

        cache_size = std::stoi(ini["default"]["cache_size"]);
        use_moe_cache = ini["default"]["use_moe_cache"] == "True" || 
                        ini["default"]["use_moe_cache"] == "1";
        
        fix_cache_size = std::stoi(ini["default"]["fix_cache_size"]);
        
        max_val = std::stof(ini["default"]["max_val"]);
        threshold = std::stof(ini["default"]["threshold"]);
        dec_in_cache = std::stof(ini["default"]["dec_in_cache"]);
        dec_out_cache = std::stof(ini["default"]["dec_out_cache"]);
        
        std::string experts_str = ini["default"]["top_k_experts"];
        
        experts_str.erase(std::remove_if(experts_str.begin(), experts_str.end(), ::isspace), experts_str.end());
        
        std::vector<std::vector<int>> experts;
        std::vector<int> current_row;
        bool inside_row = false;
        
        for (size_t i = 0; i < experts_str.length(); i++) {
            if (experts_str[i] == '[') {
                if (inside_row) {
                    current_row.clear();
                } else {
                    inside_row = true;
                }
            }
            else if (experts_str[i] == ']') {
                if (i + 1 < experts_str.length() && experts_str[i + 1] != ',') {
                    if (!current_row.empty()) {
                        experts.push_back(current_row);
                    }
                    inside_row = false;
                }
                else {
                    if (!current_row.empty()) {
                        experts.push_back(current_row);
                        current_row.clear();
                    }
                }
            }
            else if (experts_str[i] == ',') {
                continue;
            }
            else if (inside_row && isdigit(experts_str[i])) {
                size_t pos = i;
                while (pos < experts_str.length() && isdigit(experts_str[pos])) pos++;
                current_row.push_back(std::stoi(experts_str.substr(i, pos - i)));
                i = pos - 1;
            }
        }
        
        top_k_experts = experts;
    }

    std::string format_2d_vector(const std::vector<std::vector<int>>& vec) const {
        std::stringstream ss;
        ss << "[";
        for (size_t i = 0; i < vec.size(); i++) {
            ss << "[";
            for (size_t j = 0; j < vec[i].size(); j++) {
                ss << vec[i][j];
                if (j < vec[i].size() - 1) {
                    ss << ", ";
                }
            }
            ss << "]";
            if (i < vec.size() - 1) {
                ss << ", ";
            }
        }
        ss << "]";
        return ss.str();
    }

    void print() const
    {
        // TODO: replace with FT_LOG
        std::cout << "arena_size: " << arena_size << std::endl
                  << "encoder_fetcher_mode: " << int(encoder_fetcher_mode) << std::endl
                  << "decoder_fetcher_mode: " << int(decoder_fetcher_mode) << std::endl
                  << "profiling: " << profiling << std::endl
                  << "detailed_timing: " << detailed_timing << std::endl
                  << "offload_path: " << offload_path << std::endl
                  << "disk_offload: " << disk_offload << std::endl
                  << "load_from_cpp: " << load_from_cpp << std::endl
                  << "use_cache: " << use_cache << std::endl
                  << "quant_mode: " << int(quant_mode) << std::endl
                  << "vocab_size: " << vocab_size << std::endl
                  << "fetch_all: " << fetch_all << std::endl
                  << "forced_num_experts: " << forced_num_experts << std::endl
                  << "cache_policy: " << cache_policy << std::endl
                  << "cache_size: " << cache_size << std::endl
                  << "use_moe_cache: " << use_moe_cache << std::endl
                  << "fix_cache_size: " << fix_cache_size << std::endl
                  << "max_val: " << max_val << std::endl
                  << "threshold: " << threshold << std::endl
                  << "dec_in_cache: " << dec_in_cache << std::endl
                  << "dec_out_cache: " << dec_out_cache << std::endl
                  << "top_k_experts: " << format_2d_vector(top_k_experts) << std::endl;
    }


    size_t arena_size;

    FetchType encoder_fetcher_mode;
    FetchType decoder_fetcher_mode;

    bool profiling;
    bool detailed_timing;

    std::string offload_path;
    bool disk_offload;

    bool load_from_cpp;

    bool use_cache;

    QuantType quant_mode;

    int64_t vocab_size;  // workaround for missing vocab_size arg in encoder

    bool fetch_all;  // for SE-MoE

    int forced_num_experts;  // If 0, not force number of active experts

    std::string cache_policy;

    int cache_size;
    bool use_moe_cache;
    int fix_cache_size;
    float max_val;
    float threshold;
    float dec_in_cache;
    float dec_out_cache;
    std::vector<std::vector<int>> top_k_experts;

private:
    GlobalConfig()
    { 
        setDefault(); 
        if (profiling) {
            print();
        }
    }
};

}