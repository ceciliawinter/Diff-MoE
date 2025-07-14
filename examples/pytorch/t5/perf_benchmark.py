# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
perf_benchmark.py

Unlike translate_example.py, this script focuses on benchmarking the performance of the FasterTransformers T5
implementation such that it can be compared with other frameworks apples-to-apples. The changes include:

- Use random input data and disable accuracy checking.
- Use fixed input/output sequence lengths and disable early_stopping.
- Add better controls on the number of warm-ups and the number of iterations to run the inference for.

"""

import argparse
import configparser
import os
import sys
import math
from datetime import datetime
import numpy as np
import torch
import torch.distributed as dist
# dir_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(dir_path + "/../../../3rdparty/transformers/src/")

from transformers import T5ForConditionalGeneration # transformers-4.10.0-py3
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")

from examples.pytorch.t5.utils.ft_encoder import FTT5EncoderWeight, FTT5Encoder
from examples.pytorch.t5.utils.ft_decoding import FTT5DecodingWeight, FTT5Decoding, FTT5
from transformers import PreTrainedTokenizerFast
from transformers import T5ForConditionalGeneration, T5Tokenizer # transformers-4.10.0-py3
from datasets import load_dataset, Features, Value, Sequence

class TranslationResult(object):
    def __init__(self, name, frame_work):
        self.name = name
        self.frame_work = frame_work # FT or HF
        self.file_name = name + ".txt"

        self.token_list = []
        self.batch_ids_list = []
        self.batch_seq_len_list = []
        self.batch_num = 0
        self.execution_time = 0.0  # seconds
        self.token_num = 0

def true_random_tensor(low, high, size):
    num_elements = np.prod(size)
    random_bytes = os.urandom(num_elements * 8)
    random_ints = np.frombuffer(random_bytes, dtype=np.int64) % (high - low) + low
    return torch.tensor(random_ints, dtype=torch.int64).reshape(size)

class InputTokens(object):
    def __init__(self, batch_size, input_seq_len, bos_token, eos_token, vocab_size):
        # Set the last token of each sequence to eos and replace the bos/eos tokens in the middle of the sequences to
        # some other tokens.
        normal_token_list = list(range(vocab_size))
        if bos_token in normal_token_list:
            normal_token_list.remove(bos_token)
        if eos_token in normal_token_list:
            normal_token_list.remove(eos_token)
        self.input_ids = torch.randint(0, len(normal_token_list), (batch_size, input_seq_len))
        # print(f"Data type of self.input_ids: {self.input_ids[0][0].dtype}")
        # self.input_ids = true_random_tensor(0, len(normal_token_list), (batch_size, input_seq_len))
        # print(f"Data type of self.input_ids: {self.input_ids[0][0].dtype}")
        for batch_idx in range(batch_size):
            for token_idx in range(input_seq_len):
                if token_idx == input_seq_len - 1:
                    self.input_ids[batch_idx][token_idx] = eos_token
                else:
                    self.input_ids[batch_idx][token_idx] = normal_token_list[self.input_ids[batch_idx][token_idx]]
        # Set attention masks to all ones.
        self.attention_mask = torch.ones((batch_size, input_seq_len), dtype=torch.int64)

class Get_dataset(object):
    def __init__(self, dataset_name, max_length=512):
        self.max_length = max_length

        # Define CoQA dataset feature structure
        if 'coqa' in dataset_name:
            print("=" * 20 + f"{dataset_name}" + "=" * 20)
            # features = Features({
            #     "id": Value("string"),
            #     "story": Value("string"),
            #     "questions": Sequence({"input_text": Value("string")}),
            #     "answers": Sequence({"input_text": Value("string")}),
            # })
            # self.ds = load_dataset(dataset_name, features=features)
            self.ds = load_dataset('csv', data_files='/workspace/FasterTransformer/coqa/CoQA_data.csv')
        else:
            self.ds = load_dataset(dataset_name)

        available_splits = self.ds.keys()
        print(f"Available splits in dataset: {available_splits}")
        
        # Select dataset split by priority
        if 'validation' in available_splits:
            self.data = self.ds['validation']
        elif 'test' in available_splits:
            self.data = self.ds['test']
        elif 'train' in available_splits:
            self.data = self.ds['train']
        else:
            raise KeyError(f"No usable split found in dataset {dataset_name}. Available splits: {available_splits}")
        
        # Check dataset structure
        example = self.data[0]
        print("=" * 20)
        print(f"=======Dataset example structure: {example.keys()}")
        
        # Define dataset types and corresponding processing methods
        dataset_types = {
            'qa': ['id', 'title', 'context', 'question', 'answers'],
            'summary': ['document', 'summary', 'id'],
            'classification': ['text', 'sentence', 'premise', 'hypothesis'],
            'translation': ['translation'],  # Add translation type
            'coqa': ['id', 'text', 'questions', 'answers']
        }
        
        task_prefixes = {
            'qa': 'Answer the question: ',
            'summary': 'Summarize: ',
            'classification': 'Classify with 1 or 0: ',
            'translation': 'Translate to English: ',
        }
        
        # Detect dataset type
        dataset_type = None
        for dtype, fields in dataset_types.items():
            if all(field in example.keys() for field in fields):
                dataset_type = dtype
                break
            
        print(f"Detected dataset type: {dataset_type}")
        
        # Process text according to dataset type and truncate
        truncated_count = 0
        if dataset_type == 'coqa':
            self.texts = []
            for item in self.data:
                story = item["text"]
                questions = item["questions"]
                answers = item["answers"]

                n_pairs = min(len(questions), len(answers))
                if n_pairs == 0:
                    continue

                # Iterate over each Q&A pair
                for q_idx in range(n_pairs):
                    history = []
                    for prev_idx in range(max(0, q_idx - 3), q_idx):
                        history.append(f"Q: {questions[prev_idx]} A: {answers[prev_idx]}")
                    history_text = " ".join(history)

                    current_q = questions[q_idx]
                    if not current_q:
                        continue

                    input_text = (f"Context: {story}\n"
                                  f"Chat History: {history_text}\n"
                                  f"Current Question: {current_q}")

                    self.texts.append(input_text)
        elif dataset_type == 'qa':
            # Q&A dataset: combine questions and context
            self.texts = []
            for item in self.data:
                text = f"{task_prefixes['qa']}Question: {item['question']} Context: {item['context']}"
                if len(text) > self.max_length:
                    text = text[:self.max_length]
                    truncated_count += 1
                self.texts.append(text)
                
        elif dataset_type == 'summary':
            # Summary dataset: use document field
            self.texts = []
            for item in self.data:
                text = f"{task_prefixes['summary']}{item['document']}"
                if len(text) > self.max_length:
                    text = text[:self.max_length]
                    truncated_count += 1
                self.texts.append(text)
                
        elif dataset_type == 'translation':
            # Translation dataset: check available language pairs
            self.texts = []
            if isinstance(example['translation'], dict):
                available_langs = example['translation'].keys()
                src_lang = list(available_langs)[0]
                for item in self.data:
                    text = f"{task_prefixes['translation']}{item['translation'][src_lang]}"
                    if len(text) > self.max_length:
                        text = text[:self.max_length]
                        truncated_count += 1
                    self.texts.append(text)
        else:
            # Other types: try common text fields
            text_field = None
            for field in ['text', 'sentence', 'premise', 'hypothesis']:
                if field in example:
                    text_field = field
                    break
            
            if text_field is None:
                raise KeyError(f"No suitable text field found in dataset. Available fields: {example.keys()}")
            
            self.texts = []
            for item in self.data:
                text = f"{task_prefixes['classification']}{item[text_field]}"
                if len(text) > self.max_length:
                    text = text[:self.max_length]
                    truncated_count += 1
                self.texts.append(text)
        
        print(f"Successfully loaded {len(self.texts)} examples")
        if truncated_count > 0:
            print(f"Truncated {truncated_count} examples to maximum length of {self.max_length}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.texts[idx]
        return self.texts[idx]
    
def translate(args_dict):
    torch.set_printoptions(precision=6)
    batch_size = args_dict['batch_size']
    beam_size = args_dict['beam_width']
    output_seq_len = args_dict['seq_len']
    input_seq_len = args_dict['input_seq_len'] if args_dict['input_seq_len'] > 0 else output_seq_len
    time_args = args_dict["test_time"]
    beam_search_diversity_rate = args_dict['beam_search_diversity_rate']
    topk = args_dict['sampling_topk']
    topp = args_dict['sampling_topp']
    tensor_para_size = args_dict['tensor_para_size']
    pipeline_para_size = args_dict['pipeline_para_size']
    warmup_iterations = args_dict['warmup_iterations']
    infer_iterations = args_dict['iterations']
    infer_duration = args_dict['duration']
    dataset_name = args_dict['dataset_name']
    seed = args_dict['seed']
    skip_gemm = args_dict['skip_gemm']
    torch.manual_seed(seed)
    cache_size = args_dict['cache_size']
    use_moe_cache = args_dict['use_moe_cache']
    adapter_path = args_dict['adapter_path']
    layer_num = args_dict['layer_num']
    top_k_experts = args_dict['top_k_experts']
    fix_cache_size = args_dict['fix_cache_size']
    ## huggingface without bias and use relative position embedding
    ## relative position embedding -> 0, absolute position embedding -> 1
    t5_with_bias = False
    use_gated_activation = False
    t5_with_moe = False
    position_embedding_type = 0
    weight_data_type = np.float32
    ## only huggingface model path supported
    model_path = args_dict['model_path'] if args_dict['model_path'] != None else args_dict['model']
    print("model_path = ", model_path)
    ckpt_path = args_dict['ckpt_path']
    model_type = args_dict['model_type']
    ## read checkpoint config if exists
    ckpt_config = configparser.ConfigParser()
    activation_type = "relu"
    if (model_type in ["Megatron", "Megatron-DeepSpeed"]):
        ckpt_config_path = os.path.join(ckpt_path, 'config.ini')
        if os.path.isfile(ckpt_config_path):
            ckpt_config.read(ckpt_config_path)
            ## update structure config
            t5_with_bias = ckpt_config.getboolean('structure', 't5_with_bias')
            position_embedding_type = 0 if ckpt_config.get('structure', 'position_embedding_type') == 'relative' else 1
            use_gated_activation = ckpt_config.getboolean('structure', 'use_gated_activation')
            t5_with_moe= ckpt_config.getint('structure', 't5_with_moe') == 1
            weight_data_type = {"fp16": np.float16, "fp32": np.float32}[ckpt_config.get("encoder", "weight_data_type")]
            # activation_type = "gated-gelu" if use_gated_activation else "gelu" # change to gelu, which is default setting of Megatron T5
            activation_type = ckpt_config.get('encoder', 'dense_act_fn')
            moe_layers_in_encoder = []
            moe_layers_in_decoder = []
            if (ckpt_config.get('structure', 'moe_layers_in_encoder') != '[]'):
                moe_layers_in_encoder = [int(n) for n in ckpt_config.get('structure', 'moe_layers_in_encoder')[1:-1].replace(" ", "").split(',')]
                print("moe_layers_in_encoder: ", moe_layers_in_encoder)
            if (ckpt_config.get('structure', 'moe_layers_in_decoder') != '[]'):
                moe_layers_in_decoder = [int(n) for n in ckpt_config.get('structure', 'moe_layers_in_decoder')[1:-1].replace(" ", "").split(',')]

        else:
            raise Exception("config file does exist with the ckpt !")

    if model_type == "Megatron" and args_dict['ckpt_path'] == None:
        raise Exception("Megatron T5 model needs to specify checkpoint path !")

    print("\n=============== Argument ===============")
    for key in args_dict:
        print("{}: {}".format(key, args_dict[key]))
    print("========================================")

    lib_path = args_dict['lib_path']

    from peft import PeftModel
    print(f"load model from {model_path}")
    if adapter_path is not None:
        t5_model_base = T5ForConditionalGeneration.from_pretrained(model_path)
        t5_model = PeftModel.from_pretrained(t5_model_base, adapter_path)
    else:
        t5_model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    if dist.is_mpi_available():
        try:
            dist.init_process_group(backend='mpi')
            rank = dist.get_rank()
        except:
            rank = dist.get_rank()
    else:
        rank = 0
    print("rank= ",rank)
    
    if time_args.find("0") != -1 or time_args.find("2") != -1:
        t5_model = t5_model.to(rank)
        if args_dict['data_type'] == 'fp16':
            t5_model = t5_model.half()
    # model_path_adapter = f"model_path"
    if adapter_path is not None:
        tokenizer = T5Tokenizer.from_pretrained(f"{adapter_path}")
    else:
        tokenizer = T5Tokenizer.from_pretrained(f"{model_path}")

    encoder_config = t5_model.encoder.config
    decoder_config = t5_model.decoder.config
    encoder_config.update({"num_experts": 0})
    decoder_config.update({"num_experts": 0})
    encoder_config.update({"moe_layer_index": []})
    decoder_config.update({"moe_layer_index": []})

    q_scaling = 1.0 / (math.sqrt(encoder_config.d_kv))
    if (model_type in ["Megatron", "Megatron-DeepSpeed"]):
        ## update configs when using Megatron model structure
        # q_scaling = 1.0

        encoder_config.d_model = ckpt_config.getint('encoder', 'd_model')
        encoder_config.vocab_size = ckpt_config.getint('encoder', 'vocab_size')
        encoder_config.num_heads = ckpt_config.getint('encoder', 'num_heads')
        encoder_config.d_kv = ckpt_config.getint('encoder', 'd_kv')
        encoder_config.d_ff = ckpt_config.getint('encoder', 'd_ff')
        encoder_config.num_layers = ckpt_config.getint('encoder', 'num_layers')
        encoder_config.relative_attention_num_buckets = ckpt_config.getint('encoder', 'relative_attention_num_buckets_or_max_pos_seq_len')
        if model_type == "Megatron-DeepSpeed":
            encoder_config.num_experts = ckpt_config.getint('encoder', 'num_experts')
            encoder_config.moe_layer_index = moe_layers_in_encoder

        decoder_config.d_model = ckpt_config.getint('decoder', 'd_model')
        decoder_config.vocab_size = ckpt_config.getint('decoder', 'vocab_size')
        decoder_config.num_heads = ckpt_config.getint('decoder', 'num_heads')
        decoder_config.d_kv = ckpt_config.getint('decoder', 'd_kv')
        decoder_config.d_ff = ckpt_config.getint('decoder', 'd_ff')
        decoder_config.num_layers = ckpt_config.getint('decoder', 'num_layers')
        decoder_config.relative_attention_num_buckets = ckpt_config.getint('decoder', 'relative_attention_num_buckets_or_max_pos_seq_len')
        if model_type == "Megatron-DeepSpeed":
            decoder_config.num_experts = ckpt_config.getint('decoder', 'num_experts')
            decoder_config.moe_layer_index = moe_layers_in_decoder
        decoder_config.decoder_start_token_id = ckpt_config.getint('decoder', 'decoder_start_token_id')
        decoder_config.eos_token_id = ckpt_config.getint('decoder', 'eos_token_id')

    print(f"{model_type} encoder_config: {encoder_config}")
    print(f"{model_type} decoder_config: {decoder_config}")

    if os.path.isfile("gemm_config.in") and rank == 0:
        cmd = f"rm gemm_config.in"
        print(f"Run {cmd}")
        os.system(cmd)
    translation_result_list = []
    if time_args.find("0") != -1:
        translation_result_list.append(TranslationResult("hf-beamsearch-warmup", "HF"))
        translation_result_list.append(TranslationResult("hf-beamsearch", "HF"))
    if time_args.find("1") != -1:
        translation_result_list.append(TranslationResult("ft-beamsearch-warmup", "FT"))
        translation_result_list.append(TranslationResult("ft-beamsearch", "FT"))
        if rank == 0 and not skip_gemm:
            is_fp16 = 1 if args_dict['data_type'] == 'fp16' else 0
            cmd = f"./bin/t5_gemm {math.ceil(batch_size / pipeline_para_size)} {beam_size} {128} " \
                f"{encoder_config.d_model} {encoder_config.num_heads} {encoder_config.d_kv} {encoder_config.d_ff} " \
                f"{decoder_config.d_model} {decoder_config.num_heads} {decoder_config.d_kv} {decoder_config.d_ff} " \
                f"{decoder_config.vocab_size} {is_fp16} {tensor_para_size} 1 > .tmp_gemm.log"
            print(f"Run gemm test: {cmd}")
            os.system(cmd)
    if time_args.find("2") != -1:
        translation_result_list.append(TranslationResult("hf-sampling-warmup", "HF"))
        translation_result_list.append(TranslationResult("hf-sampling", "HF"))
    if time_args.find("3") != -1: # test_time = 3 select this branch
        translation_result_list.append(TranslationResult("ft-sampling-warmup", "FT"))
        translation_result_list.append(TranslationResult("ft-sampling", "FT"))
        if rank == 0 and not skip_gemm: # rank = 0, skip_gemm = False
            is_fp16 = 1 if args_dict['data_type'] == 'fp16' else 0
            cmd = f"./bin/t5_gemm {math.ceil(batch_size / pipeline_para_size)} {1} {128} " \
                f"{encoder_config.d_model} {encoder_config.num_heads} {encoder_config.d_kv} {encoder_config.d_ff} " \
                f"{decoder_config.d_model} {decoder_config.num_heads} {decoder_config.d_kv} {decoder_config.d_ff} " \
                f"{decoder_config.vocab_size} {is_fp16} {tensor_para_size} 1 > .tmp_gemm.log"
            print(f"Run gemm test: {cmd}")
            os.system(cmd)

    if time_args.find("1") != -1 or time_args.find("3") != -1:
        ft_encoder_weight = FTT5EncoderWeight(
            encoder_config,
            tensor_para_size,
            pipeline_para_size,
            t5_with_bias=t5_with_bias,
            use_gated_activation=use_gated_activation,
            t5_with_moe=t5_with_moe,
            position_embedding_type=position_embedding_type,
            weight_data_type=weight_data_type
        )
        ft_decoding_weight = FTT5DecodingWeight(
            decoder_config,
            tensor_para_size,
            pipeline_para_size,
            t5_with_bias=t5_with_bias,
            use_gated_activation=use_gated_activation,
            t5_with_moe=t5_with_moe,
            position_embedding_type=position_embedding_type,
            weight_data_type=weight_data_type,
        )

        if args_dict["ckpt_path"] is not None:
            # 31 tensors placeholders is needed for encoder
            # ft_encoder_weight.w = [torch.tensor(1, dtype=torch.float32).contiguous().cuda() for _ in range(31)]
            ft_encoder_weight.empty_weights()
            ft_decoding_weight.empty_weights()
        else:
            ft_encoder_weight.load_from_model(t5_model)
            ft_decoding_weight.load_from_model(t5_model, use_moe_cache = use_moe_cache, top_k_experts = top_k_experts)
        
        if args_dict['data_type'] == 'fp16':
            t5_model = t5_model.half()
            ft_encoder_weight.to_half()
            ft_decoding_weight.to_half()

        # This script assumes fixed sequence length, so using remove_padding will not benefit.
        remove_padding = False
        print("perf_benchmark topk = ", topk)
        ft_encoder = FTT5Encoder(ft_encoder_weight.w, lib_path, encoder_config.num_heads,
                                encoder_config.d_kv, encoder_config.d_ff,
                                encoder_config.d_model, remove_padding, encoder_config.num_layers,
                                encoder_config.relative_attention_num_buckets, encoder_config.num_experts, encoder_config.moe_layer_index,
                                128, False, q_scaling, tensor_para_size, pipeline_para_size, t5_with_bias,
                                position_embedding_type, topk, activation_type)
        ft_decoding = FTT5Decoding(ft_decoding_weight.w, lib_path,
                                decoder_config.num_heads, decoder_config.d_kv,
                                decoder_config.d_ff, encoder_config.d_model,
                                decoder_config.d_model, decoder_config.num_layers,
                                decoder_config.decoder_start_token_id,
                                # Set eos token id to -1 to effectively disable early stopping.
                                decoder_config.eos_token_id, # early-stopping TODO
                                # 1,
                                # -1,
                                decoder_config.vocab_size,
                                q_scaling,
                                decoder_config.relative_attention_num_buckets, decoder_config.num_experts, decoder_config.moe_layer_index, max_distance=128,
                                tensor_para_size=tensor_para_size, pipeline_para_size=pipeline_para_size,
                                t5_with_bias=t5_with_bias, moe_k=topk, activation_type=activation_type,
                                position_embedding_type = position_embedding_type) #, cache_size=cache_size, use_moe_cache=use_moe_cache)

        ft_t5 = FTT5(ft_encoder, ft_decoding)

    # input_token = InputTokens(batch_size, input_seq_len, decoder_config.decoder_start_token_id, decoder_config.eos_token_id, decoder_config.vocab_size)
    print("dataset_name = ", dataset_name)
    # src_text = Get_dataset(dataset_name=dataset_name)
    import pickle
    import hashlib
    def get_cached_dataset(dataset_name, max_samples=None):
        """Get dataset with caching priority"""
        # Create cache directory
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create cache filename using dataset name hash
        dataset_hash = hashlib.md5(dataset_name.encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f"{dataset_hash}_{max_samples if max_samples is not None else 'full'}.pkl")
        
        # Check if cache exists
        if os.path.exists(cache_file):
            print(f"Loading cached dataset from {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
                # Continue with normal loading if cache loading fails
        
        # No cache or cache loading failed, load dataset normally
        print(f"Loading dataset {dataset_name} and creating cache")
        dataset = Get_dataset(dataset_name)
        
        # Use full dataset if max_samples is None, otherwise limit the samples
        if max_samples is None:
            src_text = dataset[:]  # Use full dataset
            print(f"Using full dataset with {len(src_text)} samples")
        else:
            src_text = dataset[:max_samples]
            print(f"Limited dataset to {len(src_text)} samples")
        
        # Save to cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(src_text, f)
            print(f"Dataset cached to {cache_file}")
        except Exception as e:
            print(f"Error creating cache: {e}")
        
        return src_text
    print("dataset_name = ", dataset_name)
    src_text = get_cached_dataset(dataset_name)
    # src_text = Get_dataset(dataset_name)
    
    print("get dataset_name = ", dataset_name)
    
    for i in range(len(translation_result_list)): # 0 for ft-sampling-warmup, 1 for ft-sampling
        print(f"perf_benchmark.py: task{i}")
        sys.stdout.flush()
        is_warmup = (translation_result_list[i].name.find("warmup") != -1)
        min_duration = infer_duration if not is_warmup else 0
        min_iterations = infer_iterations if not is_warmup else warmup_iterations
        iter_idx = 0
        
        start_time = datetime.now()
        while iter_idx < min_iterations or (datetime.now() - start_time).total_seconds() < min_duration:
            iter_idx += 1
            print(f"perf_benchmark.py: {iter_idx}/{min_iterations}")
            sys.stdout.flush()
            prev = 0
            # start_time = datetime.now()
            print("len src_text = ", len(src_text))
            while prev < len(src_text):
                input_texts = src_text[prev:prev+batch_size]
                # print("input_texts: ", input_texts)
                prev += batch_size
                input_token = tokenizer(input_texts, return_tensors='pt', padding=True)
                print(input_token.input_ids.shape)
                # print(f"{prev}/{len(src_text)}, moe = {t5_with_moe}")
                # An example to prevent generating "Chef"
                # bad_words_text = np.array([["Chef"]]* len(input_texts), dtype=object)
                # bad_words_list = to_word_list_format(bad_words_text, tokenizer)
                # bad_words_list = torch.Tensor(bad_words_list).to(torch.int32).to("cuda").contiguous()
                bad_words_list = None

                # An example to stop generation when the model generate "Chef"
                # stop_words_text = np.array([["Chef"]] * len(input_texts), dtype=object)
                # stop_words_list = to_word_list_format(stop_words_text, tokenizer)
                # stop_words_list = torch.Tensor(stop_words_list).to(torch.int32).to("cuda").contiguous()
                stop_words_list = None

                if translation_result_list[i].frame_work == "HF":
                    if translation_result_list[i].name.find("beamsearch") != -1:
                        hf_outputs = t5_model.generate(input_token.input_ids.to("cuda"),
                                                    # min_length=output_seq_len + 1,
                                                    min_length=1,
                                                    max_length=output_seq_len + 1, # "+1" because HF counts <bos> as well.
                                                    early_stopping=True,
                                                    num_beams=beam_size)
                    elif translation_result_list[i].name.find("sampling") != -1:
                        hf_outputs = t5_model.generate(input_token.input_ids.to("cuda"),
                                                    # min_length=output_seq_len + 1,
                                                    min_length=1,
                                                    max_length=output_seq_len + 1, # "+1" because HF counts <bos> as well.
                                                    early_stopping=True,
                                                    do_sample=True,
                                                    top_k=topk if topk > 0 else None,
                                                    top_p=topp if topp > 0.0 else None)
                    translation_result_list[i].batch_ids_list.append(hf_outputs)
                    translation_result_list[i].batch_seq_len_list.append(np.ones(input_seq_len) * output_seq_len)
                elif translation_result_list[i].frame_work == "FT": # select this branch
                    tmp_beam_size = beam_size
                    import csv
                    if translation_result_list[i].name.find("sampling") != -1:
                        tmp_beam_size = 1
                        def to_word_list_format(word_dict, tokenizer):
                            flat_ids = []
                            offsets = []
                            for word_dict_item in word_dict:
                                item_flat_ids = []
                                item_offsets = []

                                words = list(csv.reader(word_dict_item))[0]
                                for word in words:
                                    ids = tokenizer.encode(word, add_special_tokens=False)

                                    if len(ids) == 0:
                                        continue

                                    item_flat_ids += ids
                                    item_offsets.append(len(ids))

                                flat_ids.append(np.array(item_flat_ids))
                                offsets.append(np.cumsum(np.array(item_offsets)))

                            pad_to = max(1, max(len(ids) for ids in flat_ids))

                            for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
                                flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
                                offsets[i] = np.pad(offs, (0, pad_to - len(offs)), constant_values=-1)

                            return np.array([flat_ids, offsets], dtype="int32").transpose((1, 0, 2))
                    stop_words_text = np.array([["Chef"]] * len(input_texts), dtype=object)
                    stop_words_list = to_word_list_format(stop_words_text, tokenizer)
                    stop_words_list = torch.Tensor(stop_words_list).to(torch.int32).to("cuda").contiguous()
                    ft_decoding_outputs, ft_decoding_seq_lens = ft_t5(input_token,
                                                                    None, # inputs_embeds
                                                                    tmp_beam_size,
                                                                    output_seq_len,
                                                                    topk,
                                                                    topp,
                                                                    beam_search_diversity_rate=beam_search_diversity_rate,
                                                                    # early_stopping=True,  
                                                                    # is_return_output_log_probs=args_dict["return_output_log_probs"],
                                                                    # is_return_cum_log_probs=args_dict["return_cum_log_probs"],
                                                                    # repetition_penalty=repetition_penalty,
                                                                    # temperature=temperature,
                                                                    # len_penalty=len_penalty,
                                                                    # bad_words_list=bad_words_list,
                                                                    stop_words_list=stop_words_list,
                                                                    )
                    translation_result_list[i].batch_ids_list.append(ft_decoding_outputs)
                    translation_result_list[i].batch_seq_len_list.append(ft_decoding_seq_lens)
                print("One round finished")
                translation_result_list[i].batch_num += 1            
        stop_time = datetime.now()
        translation_result_list[i].execution_time = (stop_time - start_time).total_seconds()
        if translation_result_list[i].name.find("warmup") != -1:
            continue
        
        for batch_token, batch_seq_len in zip(translation_result_list[i].batch_ids_list, translation_result_list[i].batch_seq_len_list):
            for j in range(len(batch_token)):
                if translation_result_list[i].frame_work == "HF":
                    translation_result_list[i].token_list.append(batch_token[j][1:])
                    translation_result_list[i].token_num += sum(batch_token[j][1:] != 0)
                elif translation_result_list[i].frame_work == "FT":
                    translation_result_list[i].token_list.append(batch_token[j][0][:batch_seq_len[j][0]])
                    translation_result_list[i].token_num += batch_seq_len[j][0]

    if rank == 0:
        for t in translation_result_list:
            if t.name.find("warmup") != -1: 
                continue
            print(f"batch_num {t.batch_num}, time {t.execution_time}")
            print(f"[INFO] {t.name} translates {t.batch_num} batches taking {t.execution_time:.2f} sec to translate "
                f"{t.token_num} tokens ({(t.execution_time / t.batch_num * 1000):.4f} ms per batch), "
                f"{(t.token_num / t.execution_time):.0f} tokens/sec.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def parse_experts_list(input_str):
        """Parse comma-separated expert list, e.g., "43,116,24,0,0,0,87,66,125,..."
        and split into 2D array based on cache_size"""
        # Handle special input
        if input_str is None or input_str.strip() == '' or input_str.strip().lower() == 'none':
            return None
        
        try:
            # Remove all whitespace characters (spaces, tabs, newlines, etc.)
            cleaned_str = ''.join(input_str.split())
            # Parse comma-separated integers
            experts = []
            for x in cleaned_str.split(','):
                if x and x.strip():  # Ensure not empty string
                    try:
                        experts.append(int(x.strip()))
                    except ValueError:
                        print(f"Warning: Skipping invalid expert value: '{x}'")
            return experts  # Return 1D array, will be split based on cache_size later
        except Exception as e:
            print(f"Warning: Could not parse experts list '{input_str}': {e}")
            return None
    
    # Regular parameters
    parser.add_argument('-batch', '--batch_size', type=int, default=1, metavar='NUMBER',
                        help='batch size (default: 1)')
    parser.add_argument('-beam', '--beam_width', type=int, default=4, metavar='NUMBER',
                        help='beam width (default: 4)')
    parser.add_argument('-s', '--seq_len', type=int, default=256, metavar='NUMBER',
                        help='fixed output sequence length, excluding bos but including eos (default: 256)')
    parser.add_argument('-inseq', '--input_seq_len', type=int, default=0, metavar='NUMBER',
                        help='fixed input sequence length, including eos (default: same as fixed output sequence length)')
    parser.add_argument('-time', '--test_time', type=str, default='', metavar='STRING',
                        help='''
                            Test the time of which one (default: '' (not test anyone) ); 
                            '': not test anyone 
                            '0': test hf_beamsearch  
                            '1': test ft_beamsearch 
                            '2': test hf_sampling 
                            '3': test ft_sampling 
                            'e.g., if you want to test tf_beamsearch and ft_sampling, 
                            then you need to use -time '03' ''')
    parser.add_argument('-diversity_rate', '--beam_search_diversity_rate', type=float, default=0.0, metavar='NUMBER',
                        help='deviersity rate of beam search. default is 0. When diversity rate = 0, it is equivalent to the naive beam search.')
    parser.add_argument('-topk', '--sampling_topk', type=int, default=1, metavar='NUMBER',
                        help='Candidate (k) value of top k sampling in decoding. Default is 1.')
    parser.add_argument('-topp', '--sampling_topp', type=float, default=0.0, metavar='NUMBER',
                        help='Probability (p) value of top p sampling in decoding. Default is 0.0. ')
    parser.add_argument('-d', '--data_type', type=str, default="fp32", metavar='STRING',
                        help='data type for inference (default: fp32)', choices=['fp32', 'fp16'])
    parser.add_argument('-ld', '--load_data_type', type=str, default="fp32", metavar='STRING',
                        help='data type for loading weights (default: fp32)', choices=['fp32', 'fp16'])
    parser.add_argument('-lib_path', '--lib_path', type=str, default="lib/libth_transformer.so", metavar='STRING',
                        help='the path of FasterTransformer pytorch t5 op library.')
    parser.add_argument('-model_path', '--model_path', type=str, default=None, metavar='STRING',
                        help='T5 model path.')
    parser.add_argument('-model', '--model', type=str, default="t5-small", metavar='STRING',
                        help='T5 model size. Only used when --model_path=None', choices=["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"])
    parser.add_argument('-adapter_path', '--adapter_path', type=str, default=None, metavar='STRING',
                        help='T5 model path.')
    parser.add_argument('-tensor_para_size', '--tensor_para_size', type=int, default=1, metavar='NUMBER',
                        help='size of tensor parallelism (default: 1)')
    parser.add_argument('-pipeline_para_size', '--pipeline_para_size', type=int, default=1, metavar='NUMBER',
                        help='size of pipeline parallelism (default: 1)')
    # assume checkpoint config is also in the same path
    parser.add_argument('--ckpt_path', type=str, help='path to the checkpoint file.')
    parser.add_argument('--model_type', type=str, default="Huggingface", choices=["Huggingface", "Megatron", "Megatron-DeepSpeed"],
                        help='Megatron T5 uses bias and supports both absulte and relative positional embedding;'
                        'Huggingface T4 adopts the paper\'s implementation and has no bias')
    # flags for performance benchmarking
    parser.add_argument('-warmup_iter', '--warmup_iterations', type=int, default=1, metavar='NUMBER',
                        help='Number of warm-up iterations for each implementation.')
    parser.add_argument('-iter', '--iterations', type=int, default=10, metavar='NUMBER',
                        help='Minimal number of inference iterations for each implementation.')
    parser.add_argument('-duration', '--duration', type=int, default=3, metavar='NUMBER',
                        help='Minimal duration in seconds for inference iterations for each implementation.')
    parser.add_argument('-seed', '--seed', type=int, default=0, metavar='NUMBER',
                        help='Random seed used to generate random input values.')
    parser.add_argument('-skip_gemm', '--skip_gemm', action="store_true",
                        help='Skip the gemm autotuning by not calling the ./bin/t5_gemm binary.')
    parser.add_argument('-dataset_name', '--dataset_name', type=str, default="gimmaru/glue-sst2", metavar='STRING',
                        help='dataset name.')
    parser.add_argument('-cache_size', '--cache_size', type=int, default=6, metavar='NUMBER',
                        help='Cache size for each expert layer')
    parser.add_argument('-use_moe_cache', '--use_moe_cache', type=bool, default=True, metavar='BOOL',
                        help='Whether to use cache for the expert priority.')
    parser.add_argument('-fix_cache_size', '--fix_cache_size', type=int, default=6, metavar='NUMBER',
                        help='Cache size for the expert priority.')
    parser.add_argument('-top_k_experts', '--top_k_experts', type=parse_experts_list, 
                        default=None,
                        metavar='LIST',
                        help='Comma-separated list of expert IDs, e.g., "43,116,24,0,0,0,87,66,125,...". Will be reshaped based on cache_size.')
    parser.add_argument('-layer_num', '--layer_num', type=int, default=12, metavar='NUMBER',
                        help='Number of transformer layers')
    parser.add_argument('-max_samples', '--max_samples', type=int, default=None, metavar='NUMBER',
                        help='Maximum number of samples to use from dataset (default: use full dataset)')
    
    # Parse arguments and handle defaults
    args = parser.parse_args()
    args_dict = vars(args)
    
    # Handle top_k_experts
    if args_dict['top_k_experts'] is None:
        # If not provided, create default 2D array
        layer_num = args_dict['layer_num']
        cache_size = args_dict['cache_size']
        if cache_size > 0:
            args_dict['top_k_experts'] = [[i for i in range(cache_size)] for _ in range(layer_num)]
            print(f"Using default top_k_experts based on layer_num={layer_num} and cache_size={cache_size}")
        else:
            # If cache_size is 0, set to empty list
            args_dict['top_k_experts'] = []
            print(f"Since cache_size={cache_size}, setting top_k_experts to empty list")
    else:
        # If 1D array is provided, split into 2D array based on cache_size
        experts_list = args_dict['top_k_experts']
        cache_size = args_dict['cache_size']
        
        # Check if cache_size is 0 to avoid division by zero
        if cache_size <= 0:
            print(f"Warning: cache_size={cache_size}, setting top_k_experts to empty list")
            args_dict['top_k_experts'] = []
        else:
            # Calculate number of complete layers
            complete_layers = len(experts_list) // cache_size
            
            # Build 2D array
            top_k_experts_2d = []
            for i in range(complete_layers):
                start_idx = i * cache_size
                layer_experts = experts_list[start_idx:start_idx + cache_size]
                top_k_experts_2d.append(layer_experts)
            
            # Handle remaining experts
            remaining = len(experts_list) % cache_size
            if remaining > 0:
                start_idx = complete_layers * cache_size
                layer_experts = experts_list[start_idx:]
                # Pad with zeros for insufficient part
                while len(layer_experts) < cache_size:
                    layer_experts.append(0)
                top_k_experts_2d.append(layer_experts)
            
            args_dict['top_k_experts'] = top_k_experts_2d
            print(f"Reshaped top_k_experts into {len(top_k_experts_2d)} layers with cache_size={cache_size}")
            
            # Print 2D array in more readable format
            print("top_k_experts = [")
            for row in top_k_experts_2d:
                print(f"    {row},")
            print("]")
    
    translate(args_dict)
