import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math


# perform qk calculation and get indices
# this version will not update in inference mode

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class PyramidKVCluster():
    """
    Prefill: PyramidKV
    decoding: None, slide, adaptive, discontinuous
    """
    
    current_decoding_step = 0
    jump_step = 0
    jump_layer = 0

    def __init__(self, decoding_metric = 'None', delta=15, num_hidden_layers = 32, decoding_window_size = 1024, decoding_recent_size = 256, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', beta = 20, num_layers = 80, layer_idx=None):
        
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        
        self.steps = -1
        self.beta = beta
        
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        ##### Add decoding window #####
        self.decoding_metric = decoding_metric
        self.decoding_window_size = decoding_window_size
        self.decoding_recent_size = decoding_recent_size
        assert self.decoding_window_size - self.decoding_recent_size > 0
        ##### Add delta #####
        self.delta = delta

    def reset(self, decoding_metric = 'None', delta=15, num_hidden_layers = 32, decoding_window_size = 1024, decoding_recent_size = 256, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        ##### Add decoding window #####
        self.decoding_metric = decoding_metric
        self.decoding_window_size = decoding_window_size
        self.decoding_recent_size = decoding_recent_size
        assert self.decoding_window_size - self.decoding_recent_size > 0
        ##### Add delta #####
        self.delta = delta
        self.num_hidden_layers = num_hidden_layers

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        # TODO
        # window_sizes = 32
        min_num = (self.max_capacity_prompt - self.window_size) // self.beta
        max_num = (self.max_capacity_prompt - self.window_size) * 2 - min_num
        
            
        if max_num >= q_len - self.window_size:
            max_num = q_len - self.window_size
            min_num = (self.max_capacity_prompt - self.window_size) * 2 - max_num
    
       
        steps = (max_num - min_num) // self.num_hidden_layers
        max_capacity_prompt = max_num - self.layer_idx * steps
        
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        elif q_len < (self.max_capacity_prompt - self.window_size) * 2:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim) # Pyramidkv(snapkv)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(max_capacity_prompt, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states
    
    def update_kv_in_decoding(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if decoding phase
        assert query_states.shape[-2]==1
        bsz, num_heads, k_len, head_dim = key_states.shape
        q_len = query_states.shape[-2]

        ##### Set decoding compress strategy #####
        if self.decoding_metric == 'None':
            return key_states, value_states
        elif self.decoding_metric == 'slide':
            decoding_window_size = self.decoding_window_size
            window_size = self.decoding_recent_size
            
            if k_len < self.max_capacity_prompt + decoding_window_size:
                return key_states, value_states
            else:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim) # bsz, num_heads, q_len=1, k_len
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_sum = attn_weights[:, :, :, : -window_size].sum(dim = -2)
                attn_cache = attn_weights_sum
                
                # prefill cache
                prefill_indices = torch.tensor(range(self.max_capacity_prompt), dtype=torch.int64).to(key_states.device)
                prefill_indices = prefill_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)

                # decoding cache
                decoding_indices = attn_cache[:, :, self.max_capacity_prompt:].topk(decoding_window_size - window_size, dim=-1).indices
                decoding_indices += self.max_capacity_prompt # add offset
                decoding_indices = decoding_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                
                # all cache
                indices = torch.cat((prefill_indices, decoding_indices), dim=2)

                k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -window_size:, :]
                v_cur = value_states[:, :, -window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                return key_states, value_states
        elif self.decoding_metric == 'adaptive':
            window_size = self.decoding_recent_size
            decoding_window_size = window_size + PyramidKVCluster.current_decoding_step//(self.delta*self.num_hidden_layers) # TODO: change the step size
            PyramidKVCluster.current_decoding_step += 1
            
            if k_len < self.max_capacity_prompt + decoding_window_size:
                return key_states, value_states
            else:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_sum = attn_weights[:, :, :, : -window_size].sum(dim = -2)
                attn_cache = attn_weights_sum
                
                # prefill cache
                prefill_indices = torch.tensor(range(self.max_capacity_prompt), dtype=torch.int64).to(key_states.device)
                prefill_indices = prefill_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)

                # decoding cache
                decoding_indices = attn_cache[:, :, self.max_capacity_prompt:].topk(decoding_window_size - window_size, dim=-1).indices
                decoding_indices += self.max_capacity_prompt # add offset
                decoding_indices = decoding_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                
                # all cache
                indices = torch.cat((prefill_indices, decoding_indices), dim=2)

                k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -window_size:, :]
                v_cur = value_states[:, :, -window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                return key_states, value_states
        elif self.decoding_metric == 'discontinuous':
            window_size = self.decoding_recent_size
            decoding_window_size = window_size + PyramidKVCluster.current_decoding_step//(self.delta*self.num_hidden_layers) # TODO: change the step size
            PyramidKVCluster.current_decoding_step += 1

            
            if k_len < self.max_capacity_prompt + decoding_window_size:
                return key_states, value_states
            elif PyramidKVCluster.jump_step < self.delta*self.num_hidden_layers:
                PyramidKVCluster.jump_step += 1
                return key_states, value_states
            else:
                PyramidKVCluster.jump_layer += 1
                if PyramidKVCluster.jump_layer == self.num_hidden_layers:
                    PyramidKVCluster.jump_step = 0
                    PyramidKVCluster.jump_layer = 0
                
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_sum = attn_weights[:, :, :, : -window_size].sum(dim = -2)
                attn_cache = attn_weights_sum
                
                # prefill cache
                prefill_indices = torch.tensor(range(self.max_capacity_prompt), dtype=torch.int64).to(key_states.device)
                prefill_indices = prefill_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)

                # decoding cache
                decoding_indices = attn_cache[:, :, self.max_capacity_prompt:].topk(decoding_window_size - window_size, dim=-1).indices
                decoding_indices += self.max_capacity_prompt # add offset
                decoding_indices = decoding_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                
                # all cache
                indices = torch.cat((prefill_indices, decoding_indices), dim=2)

                k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -window_size:, :]
                v_cur = value_states[:, :, -window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                return key_states, value_states
        else:
            raise ValueError('Decoding metric not supported')


class PyramidInferCluster():
    """
    Prefill: PyramidInfer
    decoding: None, pyramidinfer
    """

    def __init__(self, decoding_metric = 'None', num_hidden_layers = 32, decoding_window_size = 1024, decoding_recent_size = 256, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', beta = 20, num_layers = 80, layer_idx=None):
        
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        
        self.steps = -1
        self.beta = beta
        
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        ##### Add decoding window #####
        self.decoding_metric = decoding_metric
        self.decoding_window_size = decoding_window_size
        self.decoding_recent_size = decoding_recent_size
        assert self.decoding_window_size - self.decoding_recent_size > 0

    def reset(self, decoding_metric = 'None', decoding_window_size = 1024, decoding_recent_size = 256, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        ##### Add decoding window #####
        self.decoding_metric = decoding_metric
        self.decoding_window_size = decoding_window_size
        self.decoding_recent_size = decoding_recent_size
        assert self.decoding_window_size - self.decoding_recent_size > 0

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        # TODO
        # window_sizes = 32
        min_num = (self.max_capacity_prompt - self.window_size) // self.beta
        max_num = (self.max_capacity_prompt - self.window_size) * 2 - min_num
        
            
        if max_num >= q_len - self.window_size:
            max_num = q_len - self.window_size
            min_num = (self.max_capacity_prompt - self.window_size) * 2 - max_num
    
       
        steps = (max_num - min_num) // self.num_hidden_layers
        max_capacity_prompt = max_num - self.layer_idx * steps
        
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        elif q_len < (self.max_capacity_prompt - self.window_size) * 2:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, :, : -self.window_size].sum(dim = -2)
            attn_cache = attn_weights_sum
            indices = attn_cache.topk(self.max_capacity_prompt, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, :, : -self.window_size].sum(dim = -2)
            attn_cache = attn_weights_sum
            indices = attn_cache.topk(max_capacity_prompt, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states
    
    def update_kv_in_decoding(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if decoding phase
        assert query_states.shape[-2]==1
        bsz, num_heads, k_len, head_dim = key_states.shape
        q_len = query_states.shape[-2]

        ##### Set decoding compress strategy #####
        if self.decoding_metric == 'None':
            return key_states, value_states
        elif self.decoding_metric == 'pyramidinfer':
            # prefill+decoding cache, compute the number of tokens to keep in the cache
            decoding_window_size = self.decoding_window_size
            window_size = self.decoding_recent_size
            min_num = (self.max_capacity_prompt + decoding_window_size - window_size) // 2 # TODO beta set to 2
            max_num = (self.max_capacity_prompt + decoding_window_size - window_size) * 2 - min_num

            steps = (max_num - min_num) // self.num_hidden_layers
            max_capacity_prompt = max_num - self.layer_idx * steps

            if k_len < self.max_capacity_prompt + decoding_window_size:
                return key_states, value_states
            elif k_len < (self.max_capacity_prompt - window_size) * 2 + decoding_window_size:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_sum = attn_weights[:, :, :, : -window_size].sum(dim = -2)
                attn_cache = attn_weights_sum
                # decoding cache
                decoding_indices = attn_cache.topk(self.max_capacity_prompt + decoding_window_size - window_size, dim=-1).indices
                decoding_indices = decoding_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                indices = decoding_indices
                k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -window_size:, :]
                v_cur = value_states[:, :, -window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                return key_states, value_states
            else:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_sum = attn_weights[:, :, :, : -window_size].sum(dim = -2)
                attn_cache = attn_weights_sum
                # decoding cache
                decoding_indices = attn_cache.topk(max_capacity_prompt + decoding_window_size, dim=-1).indices
                decoding_indices = decoding_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                indices = decoding_indices
                k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -window_size:, :]
                v_cur = value_states[:, :, -window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                return key_states, value_states
        else:
            raise ValueError('Decoding metric not supported')


class SnapKVCluster():
    """
    Prefill: SnapKV
    decoding: None, slide, adaptive, discontinuous
    """

    current_decoding_step = 0
    jump_step = 0
    jump_layer = 0

    def __init__(self, decoding_metric = 'None', delta=15, num_hidden_layers = 32, decoding_window_size = 1024, decoding_recent_size = 256, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        ##### Add decoding window #####
        self.decoding_metric = decoding_metric
        self.decoding_window_size = decoding_window_size
        self.decoding_recent_size = decoding_recent_size
        assert self.decoding_window_size - self.decoding_recent_size > 0
        ##### Add delta #####
        self.delta = delta
        ##### Add num_hidden_layers #####
        self.num_hidden_layers = num_hidden_layers

    def reset(self, decoding_metric = 'None', delta=15, num_hidden_layers = 32, decoding_window_size = 1024, decoding_recent_size = 256, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        ##### Add decoding window #####
        self.decoding_metric = decoding_metric
        self.decoding_window_size = decoding_window_size
        self.decoding_recent_size = decoding_recent_size
        assert self.decoding_window_size - self.decoding_recent_size > 0
        ##### Add delta #####
        self.delta = delta
        ##### Add num_hidden_layers #####
        self.num_hidden_layers = num_hidden_layers

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states
    
    def update_kv_in_decoding(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if decoding phase
        assert query_states.shape[-2]==1
        bsz, num_heads, k_len, head_dim = key_states.shape
        q_len = query_states.shape[-2]

        ##### Set decoding compress strategy #####
        if self.decoding_metric == 'None':
            return key_states, value_states
        elif self.decoding_metric == 'slide':
            decoding_window_size = self.decoding_window_size
            window_size = self.decoding_recent_size
            
            if k_len < self.max_capacity_prompt + decoding_window_size:
                return key_states, value_states
            else:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim) # bsz, num_heads, q_len=1, k_len
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_sum = attn_weights[:, :, :, : -window_size].sum(dim = -2)
                attn_cache = attn_weights_sum
                
                # prefill cache
                prefill_indices = torch.tensor(range(self.max_capacity_prompt), dtype=torch.int64).to(key_states.device)
                prefill_indices = prefill_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)

                # decoding cache
                decoding_indices = attn_cache[:, :, self.max_capacity_prompt:].topk(decoding_window_size - window_size, dim=-1).indices
                decoding_indices += self.max_capacity_prompt # add offset
                decoding_indices = decoding_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                
                # all cache
                indices = torch.cat((prefill_indices, decoding_indices), dim=2)

                k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -window_size:, :]
                v_cur = value_states[:, :, -window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                return key_states, value_states
        elif self.decoding_metric == 'adaptive':
            window_size = self.decoding_recent_size
            decoding_window_size = window_size + SnapKVCluster.current_decoding_step//(self.delta*self.num_hidden_layers) # TODO: change the step size
            SnapKVCluster.current_decoding_step += 1
            
            if k_len < self.max_capacity_prompt + decoding_window_size:
                return key_states, value_states
            else:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_sum = attn_weights[:, :, :, : -window_size].sum(dim = -2)
                attn_cache = attn_weights_sum
                
                # prefill cache
                prefill_indices = torch.tensor(range(self.max_capacity_prompt), dtype=torch.int64).to(key_states.device)
                prefill_indices = prefill_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)

                # decoding cache
                decoding_indices = attn_cache[:, :, self.max_capacity_prompt:].topk(decoding_window_size - window_size, dim=-1).indices
                decoding_indices += self.max_capacity_prompt # add offset
                decoding_indices = decoding_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                
                # all cache
                indices = torch.cat((prefill_indices, decoding_indices), dim=2)

                k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -window_size:, :]
                v_cur = value_states[:, :, -window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                return key_states, value_states
        elif self.decoding_metric == 'discontinuous':
            window_size = self.decoding_recent_size
            decoding_window_size = window_size + SnapKVCluster.current_decoding_step//(self.delta*self.num_hidden_layers) # TODO: change the step size
            SnapKVCluster.current_decoding_step += 1
            
            if k_len < self.max_capacity_prompt + decoding_window_size:
                return key_states, value_states
            elif SnapKVCluster.jump_step < self.delta*self.num_hidden_layers:
                SnapKVCluster.jump_step += 1
                return key_states, value_states
            else:
                SnapKVCluster.jump_layer += 1
                if SnapKVCluster.jump_layer == self.num_hidden_layers:
                    SnapKVCluster.jump_step = 0
                    SnapKVCluster.jump_layer = 0
                
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_sum = attn_weights[:, :, :, : -window_size].sum(dim = -2)
                attn_cache = attn_weights_sum
                
                # prefill cache
                prefill_indices = torch.tensor(range(self.max_capacity_prompt), dtype=torch.int64).to(key_states.device)
                prefill_indices = prefill_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)

                # decoding cache
                decoding_indices = attn_cache[:, :, self.max_capacity_prompt:].topk(decoding_window_size - window_size, dim=-1).indices
                decoding_indices += self.max_capacity_prompt # add offset
                decoding_indices = decoding_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                
                # all cache
                indices = torch.cat((prefill_indices, decoding_indices), dim=2)

                k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -window_size:, :]
                v_cur = value_states[:, :, -window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                return key_states, value_states
        else:
            raise ValueError('Decoding metric not supported')


class H2OKVCluster():
    """
    Prefill: H2O
    decoding: None, slide, adaptive, discontinuous
    """

    current_decoding_step = 0
    jump_step = 0
    jump_layer = 0

    def __init__(self, decoding_metric = 'None', delta=15, num_hidden_layers = 32, decoding_window_size = 1024, decoding_recent_size = 256, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        ##### Add decoding window #####
        self.decoding_metric = decoding_metric
        self.decoding_window_size = decoding_window_size
        self.decoding_recent_size = decoding_recent_size
        assert self.decoding_window_size - self.decoding_recent_size > 0
        ##### Add delta #####
        self.delta = delta
        ##### Add num_hidden_layers #####
        self.num_hidden_layers = num_hidden_layers

    def reset(self, decoding_metric = 'None', delta=15, num_hidden_layers = 32, decoding_window_size = 1024, decoding_recent_size = 256, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        ##### Add decoding window #####
        self.decoding_metric = decoding_metric
        self.decoding_window_size = decoding_window_size
        self.decoding_recent_size = decoding_recent_size
        assert self.decoding_window_size - self.decoding_recent_size > 0
        ##### Add delta #####
        self.delta = delta
        ##### Add num_hidden_layers #####
        self.num_hidden_layers = num_hidden_layers

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        # reset decoding step
        H2OKVCluster.current_decoding_step = 0
        H2OKVCluster.jump_step = 0
        
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, :, : -self.window_size].sum(dim = -2)
            attn_cache = attn_weights_sum
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states
    
    def update_kv_in_decoding(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if decoding phase
        assert query_states.shape[-2]==1
        bsz, num_heads, k_len, head_dim = key_states.shape
        q_len = query_states.shape[-2]

        ##### Set decoding compress strategy #####
        if self.decoding_metric == 'None':
            return key_states, value_states
        elif self.decoding_metric == 'h2o':
            decoding_window_size = self.decoding_window_size
            window_size = self.decoding_recent_size
            
            if k_len < self.max_capacity_prompt + decoding_window_size:
                return key_states, value_states
            else:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_sum = attn_weights[:, :, :, : -window_size].sum(dim = -2)
                attn_cache = attn_weights_sum
                # decoding cache
                decoding_indices = attn_cache.topk(self.max_capacity_prompt + decoding_window_size - window_size, dim=-1).indices
                decoding_indices = decoding_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                indices = decoding_indices
                k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -window_size:, :]
                v_cur = value_states[:, :, -window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                return key_states, value_states
        elif self.decoding_metric == 'slide':
            decoding_window_size = self.decoding_window_size
            window_size = self.decoding_recent_size
            
            if k_len < self.max_capacity_prompt + decoding_window_size:
                return key_states, value_states
            else:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim) # bsz, num_heads, q_len=1, k_len
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_sum = attn_weights[:, :, :, : -window_size].sum(dim = -2)
                attn_cache = attn_weights_sum
                
                # prefill cache
                prefill_indices = torch.tensor(range(self.max_capacity_prompt), dtype=torch.int64).to(key_states.device)
                prefill_indices = prefill_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)

                # decoding cache
                decoding_indices = attn_cache[:, :, self.max_capacity_prompt:].topk(decoding_window_size - window_size, dim=-1).indices
                decoding_indices += self.max_capacity_prompt # add offset
                decoding_indices = decoding_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                
                # all cache
                indices = torch.cat((prefill_indices, decoding_indices), dim=2)

                k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -window_size:, :]
                v_cur = value_states[:, :, -window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                return key_states, value_states
        elif self.decoding_metric == 'adaptive':
            window_size = self.decoding_recent_size
            decoding_window_size = window_size + H2OKVCluster.current_decoding_step//(self.delta*self.num_hidden_layers) # TODO: change the step size
            H2OKVCluster.current_decoding_step += 1
            
            if k_len < self.max_capacity_prompt + decoding_window_size:
                return key_states, value_states
            else:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_sum = attn_weights[:, :, :, : -window_size].sum(dim = -2)
                attn_cache = attn_weights_sum
                
                # prefill cache
                prefill_indices = torch.tensor(range(self.max_capacity_prompt), dtype=torch.int64).to(key_states.device)
                prefill_indices = prefill_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)

                # decoding cache
                decoding_indices = attn_cache[:, :, self.max_capacity_prompt:].topk(decoding_window_size - window_size, dim=-1).indices
                decoding_indices += self.max_capacity_prompt # add offset
                decoding_indices = decoding_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                
                # all cache
                indices = torch.cat((prefill_indices, decoding_indices), dim=2)

                k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -window_size:, :]
                v_cur = value_states[:, :, -window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                return key_states, value_states
        elif self.decoding_metric == 'discontinuous':
            window_size = self.decoding_recent_size
            decoding_window_size = window_size + H2OKVCluster.current_decoding_step//(self.delta*self.num_hidden_layers) # TODO: change the step size
            H2OKVCluster.current_decoding_step += 1
            
            if k_len < self.max_capacity_prompt + decoding_window_size:
                return key_states, value_states
            elif H2OKVCluster.jump_step < self.delta*self.num_hidden_layers:
                H2OKVCluster.jump_step += 1
                return key_states, value_states
            else:
                H2OKVCluster.jump_layer += 1
                if H2OKVCluster.jump_layer == self.num_hidden_layers:
                    H2OKVCluster.jump_step = 0
                    H2OKVCluster.jump_layer = 0
                
                
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_sum = attn_weights[:, :, :, : -window_size].sum(dim = -2)
                attn_cache = attn_weights_sum
                
                # prefill cache
                prefill_indices = torch.tensor(range(self.max_capacity_prompt), dtype=torch.int64).to(key_states.device)
                prefill_indices = prefill_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)

                # decoding cache
                decoding_indices = attn_cache[:, :, self.max_capacity_prompt:].topk(decoding_window_size - window_size, dim=-1).indices
                decoding_indices += self.max_capacity_prompt # add offset
                decoding_indices = decoding_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                
                # all cache
                indices = torch.cat((prefill_indices, decoding_indices), dim=2)

                k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -window_size:, :]
                v_cur = value_states[:, :, -window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                return key_states, value_states
        else:
            raise ValueError('Decoding metric not supported')


class StreamingLLMKVCluster():
    """
    Prefill: StreamingLLM
    decoding: None, slm
    """

    def __init__(self, decoding_metric = 'None', decoding_window_size = 1024, decoding_recent_size = 256, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        ##### Add decoding window #####
        self.decoding_metric = decoding_metric
        self.decoding_window_size = decoding_window_size
        self.decoding_recent_size = decoding_recent_size
        assert self.decoding_window_size - self.decoding_recent_size > 0

    def reset(self, decoding_metric = 'None', decoding_window_size = 1024, decoding_recent_size = 256, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        ##### Add decoding window #####
        self.decoding_metric = decoding_metric
        self.decoding_window_size = decoding_window_size
        self.decoding_recent_size = decoding_recent_size
        assert self.decoding_window_size - self.decoding_recent_size > 0

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:    
            indices = torch.tensor(range(self.max_capacity_prompt - self.window_size), dtype=torch.int64).to(key_states.device)
            indices = indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states
        
    def update_kv_in_decoding(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if decoding phase
        assert query_states.shape[-2]==1
        bsz, num_heads, k_len, head_dim = key_states.shape
        q_len = query_states.shape[-2]

        ##### Set decoding compress strategy #####
        if self.decoding_metric == 'None':
            return key_states, value_states
        elif self.decoding_metric == 'slm':
            if k_len < self.max_capacity_prompt + self.decoding_window_size:
                return key_states, value_states
            else:
                decoding_window_size = self.decoding_window_size
                window_size = self.decoding_recent_size
                
                # decoding cache
                decoding_indices = torch.tensor(range(self.max_capacity_prompt+decoding_window_size-window_size), dtype=torch.int64).to(key_states.device)
                decoding_indices = decoding_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)
                
                indices = decoding_indices

                k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -window_size:, :]
                v_cur = value_states[:, :, -window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                return key_states, value_states
        else:
            raise ValueError('Decoding metric not supported')

        
class ALLKVCluster():
    """
    Prefill: ALLKV
    decoding: None, slide, adaptive, discontinuous
    """

    allkv_max_capacity_prompt = 0
    current_decoding_step = 0
    jump_step = 0
    jump_layer = 0

    def __init__(self, decoding_metric = 'None', delta=15, num_hidden_layers = 32, decoding_window_size = 1024, decoding_recent_size = 256):
        ##### Add decoding window #####
        self.decoding_metric = decoding_metric
        self.decoding_window_size = decoding_window_size
        self.decoding_recent_size = decoding_recent_size
        assert self.decoding_window_size - self.decoding_recent_size > 0
        ##### Add delta #####
        self.delta = delta
        ##### Add num_hidden_layers #####
        self.num_hidden_layers = num_hidden_layers            

    def reset(self, decoding_metric = 'None', delta=15, num_hidden_layers = 32, decoding_window_size = 1024, decoding_recent_size = 256):
        ##### Add decoding window #####
        self.decoding_metric = decoding_metric
        self.decoding_window_size = decoding_window_size
        self.decoding_recent_size = decoding_recent_size
        assert self.decoding_window_size - self.decoding_recent_size > 0
        ##### Add delta #####
        self.delta = delta
        ##### Add num_hidden_layers #####
        self.num_hidden_layers = num_hidden_layers   
    
    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        # print(f"ALLKV: prefill not compressed")

        ##### Record max_capacity_prompt #####
        ALLKVCluster.max_capacity_prompt = key_states.shape[-2]

        # reset decoding step
        ALLKVCluster.current_decoding_step = 0
        ALLKVCluster.jump_step = 0

        return key_states, value_states
    
    def update_kv_in_decoding(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if decoding phase
        assert query_states.shape[-2]==1
        bsz, num_heads, k_len, head_dim = key_states.shape
        q_len = query_states.shape[-2]

        ##### Set decoding compress strategy #####
        if self.decoding_metric == 'None':
            return key_states, value_states
        elif self.decoding_metric == 'slide':
            decoding_window_size = self.decoding_window_size
            window_size = self.decoding_recent_size
            
            if k_len < ALLKVCluster.max_capacity_prompt + decoding_window_size:
                return key_states, value_states
            else:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_sum = attn_weights[:, :, :, : -window_size].sum(dim = -2)
                attn_cache = attn_weights_sum
                
                # prefill cache
                prefill_indices = torch.tensor(range(ALLKVCluster.max_capacity_prompt), dtype=torch.int64).to(key_states.device)
                prefill_indices = prefill_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)

                # decoding cache
                decoding_indices = attn_cache[:, :, ALLKVCluster.max_capacity_prompt:].topk(decoding_window_size - window_size, dim=-1).indices
                decoding_indices += ALLKVCluster.max_capacity_prompt # add offset
                decoding_indices = decoding_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                
                indices = torch.cat((prefill_indices, decoding_indices), dim=2)

                k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -window_size:, :]
                v_cur = value_states[:, :, -window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                return key_states, value_states
        elif self.decoding_metric == 'adaptive':
            window_size = self.decoding_recent_size
            decoding_window_size = window_size + ALLKVCluster.current_decoding_step//(self.delta*self.num_hidden_layers) # TODO: change the step size
            ALLKVCluster.current_decoding_step += 1
            
            if k_len < ALLKVCluster.max_capacity_prompt + decoding_window_size:
                return key_states, value_states
            else:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_sum = attn_weights[:, :, :, : -window_size].sum(dim = -2)
                attn_cache = attn_weights_sum
                
                # prefill cache
                prefill_indices = torch.tensor(range(ALLKVCluster.max_capacity_prompt), dtype=torch.int64).to(key_states.device)
                prefill_indices = prefill_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)

                # decoding cache
                decoding_indices = attn_cache[:, :, ALLKVCluster.max_capacity_prompt:].topk(decoding_window_size - window_size, dim=-1).indices
                decoding_indices += ALLKVCluster.max_capacity_prompt # add offset
                decoding_indices = decoding_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                
                # all cache
                indices = torch.cat((prefill_indices, decoding_indices), dim=2)

                k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -window_size:, :]
                v_cur = value_states[:, :, -window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                return key_states, value_states
        elif self.decoding_metric == 'discontinuous':
            window_size = self.decoding_recent_size
            decoding_window_size = window_size + ALLKVCluster.current_decoding_step//(self.delta*self.num_hidden_layers) # TODO: change the step size
            ALLKVCluster.current_decoding_step += 1
            
            if k_len < ALLKVCluster.max_capacity_prompt + decoding_window_size:
                return key_states, value_states
            elif ALLKVCluster.jump_step < self.delta*self.num_hidden_layers:
                ALLKVCluster.jump_step += 1
                return key_states, value_states
            else:
                ALLKVCluster.jump_layer += 1
                if ALLKVCluster.jump_layer == self.num_hidden_layers:
                    ALLKVCluster.jump_step = 0
                    ALLKVCluster.jump_layer = 0
                
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_sum = attn_weights[:, :, :, : -window_size].sum(dim = -2)
                attn_cache = attn_weights_sum
                
                # prefill cache
                prefill_indices = torch.tensor(range(ALLKVCluster.max_capacity_prompt), dtype=torch.int64).to(key_states.device)
                prefill_indices = prefill_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)

                # decoding cache
                decoding_indices = attn_cache[:, :, ALLKVCluster.max_capacity_prompt:].topk(decoding_window_size - window_size, dim=-1).indices
                decoding_indices += ALLKVCluster.max_capacity_prompt # add offset
                decoding_indices = decoding_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                
                # all cache
                indices = torch.cat((prefill_indices, decoding_indices), dim=2)

                k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -window_size:, :]
                v_cur = value_states[:, :, -window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                return key_states, value_states
        else:
            raise ValueError('Decoding metric not supported')


def init_pyramidkv(self, num_hidden_layers):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'decoding_metric'):
            self.config.decoding_metric = 'None'
        if not hasattr(self.config, 'decoding_window_size'):
            self.config.decoding_window_size = 1024
        if not hasattr(self.config, 'decoding_recent_size'):
            self.config.decoding_recent_size = 256
    
    
    self.kv_cluster = PyramidKVCluster( 
        num_hidden_layers = num_hidden_layers,
        layer_idx = self.layer_idx,
        decoding_window_size=self.config.decoding_window_size,
        decoding_recent_size=self.config.decoding_recent_size,
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        decoding_metric=self.config.decoding_metric
        )
 
def init_pyramidinfer(self, num_hidden_layers):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'decoding_metric'):
            self.config.decoding_metric = 'None'
        if not hasattr(self.config, 'decoding_window_size'):
            self.config.decoding_window_size = 1024
        if not hasattr(self.config, 'decoding_recent_size'):
            self.config.decoding_recent_size = 256
    
    
    self.kv_cluster = PyramidInferCluster( 
        num_hidden_layers = num_hidden_layers,
        layer_idx = self.layer_idx,
        decoding_window_size=self.config.decoding_window_size,
        decoding_recent_size=self.config.decoding_recent_size,
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        decoding_metric=self.config.decoding_metric
        )

def init_snapkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 4096
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'decoding_metric'):
            self.config.decoding_metric = 'None'
        if not hasattr(self.config, 'decoding_window_size'):
            self.config.decoding_window_size = 1024
        if not hasattr(self.config, 'decoding_recent_size'):
            self.config.decoding_recent_size = 256
    
    self.kv_cluster = SnapKVCluster( 
        decoding_window_size=self.config.decoding_window_size,
        decoding_recent_size=self.config.decoding_recent_size,
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        decoding_metric=self.config.decoding_metric
        )

def init_H2O(self, num_hidden_layers):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'decoding_metric'):
            self.config.decoding_metric = 'None'
        if not hasattr(self.config, 'decoding_window_size'):
            self.config.decoding_window_size = 1024
        if not hasattr(self.config, 'decoding_recent_size'):
            self.config.decoding_recent_size = 256
        if not hasattr(self.config, 'delta'):
            self.config.delta = 15
    
    
    self.kv_cluster = H2OKVCluster(
        num_hidden_layers = num_hidden_layers,
        delta=self.config.delta,
        decoding_window_size=self.config.decoding_window_size,
        decoding_recent_size=self.config.decoding_recent_size,
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        decoding_metric=self.config.decoding_metric
        )

def init_StreamingLLM(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'decoding_metric'):
            self.config.decoding_metric = 'None'
        if not hasattr(self.config, 'decoding_window_size'):
            self.config.decoding_window_size = 1024
        if not hasattr(self.config, 'decoding_recent_size'):
            self.config.decoding_recent_size = 256
    
    
    self.kv_cluster = StreamingLLMKVCluster(
        decoding_window_size=self.config.decoding_window_size,
        decoding_recent_size=self.config.decoding_recent_size,
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        decoding_metric=self.config.decoding_metric
        )
    
def init_ALLKV(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'decoding_metric'):
            self.config.decoding_metric = 'None'
        if not hasattr(self.config, 'decoding_window_size'):
            self.config.decoding_window_size = 1024
        if not hasattr(self.config, 'decoding_recent_size'):
            self.config.decoding_recent_size = 256
    
    
    self.kv_cluster = ALLKVCluster(
        decoding_metric=self.config.decoding_metric,
        decoding_window_size=self.config.decoding_window_size,
        decoding_recent_size=self.config.decoding_recent_size,
        )
