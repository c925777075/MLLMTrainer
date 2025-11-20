import torch
import time
import torch.nn.functional as F
from typing import Optional, Tuple, Union
# 确保 transformers 库已安装，FlexAttention 的集成依赖于此
from transformers.integrations.flex_attention import repeat_kv, compile_friendly_flex_attention
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.attention.flex_attention import BlockMask, flex_attention # flex_attention 实际未直接使用，但 BlockMask 是关键

Offset = Union

def flex_attention_mask(doc_ids, window_mask=True, stride_mask=True):
    """
    为 SDPA 生成完整的注意力掩码。
    此函数生成一个 (batch_size, seq_len, seq_len) 的布尔掩码，
    其中 True 表示允许注意力，False 表示禁止。
    """
    if not window_mask and not stride_mask:
        raise ValueError("At least one of 'window_mask' or 'stride_mask' must be enabled.")
    if not isinstance(doc_ids, torch.Tensor):
        doc_ids = torch.tensor(doc_ids, dtype=torch.long)

    batch_size, seq_len = doc_ids.shape
    device = doc_ids.device

    # 1. 初始化基础掩码（全False表示不可见）
    mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool, device=device)

    # 2. 遍历batch中的每个样本
    for b in range(batch_size):
        sample = doc_ids[b]

        # 获取非padding文档的边界（忽略0）
        boundaries = []
        i = 0
        while i < seq_len:
            if sample[i] == 0:  # 跳过padding
                i += 1
                continue

            doc_id = sample[i]
            start = i
            while i < seq_len and sample[i] == doc_id:
                i += 1
            end = i - 1
            boundaries.append((start, end))

        # 3. 构建文档内部因果注意力
        if window_mask:
            for start, end in boundaries:
                doc_len = end - start + 1
                doc_mask = torch.tril(torch.ones(doc_len, doc_len, dtype=torch.bool, device=device))
                mask[b, start:end+1, start:end+1] = doc_mask

        # 4. 修改：让最后一个文档的所有token都能看到前面所有文档的最后一个token
        if stride_mask and boundaries:
            # 收集所有文档的最后一个token位置
            last_tokens = [end for _, end in boundaries]
            num_docs = len(last_tokens)
            if num_docs >= 2:
                # 获取最后一个文档的起止位置
                last_doc_start, last_doc_end = boundaries[-1]

                # 前面所有文档的最后一个token位置（排除最后一个文档自身）
                prev_lasts = last_tokens[:-1]

                # 让最后一个文档的所有token都能看到前面所有文档的最后一个token
                mask[b, last_doc_start:last_doc_end+1, prev_lasts] = True

                # 保留原始逻辑：除最后一个文档外，其他文档的最后一个token也能看到前面所有文档的最后一个token
                for idx in range(1, num_docs - 1):
                    current_last = last_tokens[idx]
                    mask[b, current_last, prev_lasts[:idx]] = True
    # 5. 处理padding位置
    padding_mask = (doc_ids == 0)
    mask = mask & ~padding_mask.unsqueeze(1)  # 行掩码：查询是padding则不关注任何键
    mask = mask & ~padding_mask.unsqueeze(2)  # 列掩码：键是padding则不被任何查询关注

    return mask

def prepare_batch_doc_info_optimized(batch_doc_ids: torch.Tensor):
    """
    预处理批次文档ID序列，提取每个Token的文档信息，针对批量输入进行优化。

    Args:
        batch_doc_ids: 一个二维的torch.LongTensor，形状为 (batch_size, sequence_length)，
                       其中 0 代表 padding id。

    Returns:
        tuple:
            - batch_token_to_doc_map (torch.Tensor): 形状为 (batch_size, sequence_length)，
              与输入相同，表示每个token的文档ID。
            - batch_last_token_indices_per_doc (torch.Tensor): 形状为 (batch_size, sequence_length)，
              对于每个token，如果它是其所在文档的最后一个有效token，则存储其全局索引；
              否则为 -1。
            - batch_last_doc_start_end_indices (torch.Tensor): 形状为 (batch_size, 2)，
              存储批次中每个样本的最后一个有效文档的起始和结束的全局索引。
              如果样本全为 padding，则为 [-1, -1]。
    """
    batch_size, seq_len = batch_doc_ids.shape

    # token_to_doc_map 直接就是输入的 batch_doc_ids
    batch_token_to_doc_map = batch_doc_ids

    # --- 计算 batch_last_token_indices_per_doc ---
    # 通过比较当前token的doc_id和下一个token的doc_id来判断是否是文档的最后一个token
    next_doc_ids = torch.roll(batch_doc_ids, shifts=-1, dims=1)
    next_doc_ids[:, -1] = 0 # 标记序列末尾为0，方便后续与batch_doc_ids!=0结合判断

    # 判断每个token是否是其文档的最后一个token (非padding且与下一个doc_id不同)
    is_last_token_in_doc = (batch_token_to_doc_map!= 0) & \
                           (batch_token_to_doc_map!= next_doc_ids)

    global_indices = torch.arange(seq_len, device=batch_doc_ids.device).unsqueeze(0).expand(batch_size, -1)

    # 记录每个token是否是其文档的最后一个token的全局索引，否则为-1
    batch_last_token_indices_per_doc = torch.where(
        is_last_token_in_doc,
        global_indices,
        torch.full_like(batch_doc_ids, -1, dtype=torch.long)
    )

    # --- 计算 batch_last_doc_start_end_indices ---
    # 1. 找到每个样本中最后一个非零文档ID的索引 (即最后一个有效token的全局索引)
    is_valid_doc_id_mask = (batch_doc_ids!= 0)
    # 翻转掩码，找到最后一个True的索引
    flipped_is_valid = torch.flip(is_valid_doc_id_mask, dims=[1])
    last_valid_token_in_seq_flipped = torch.argmax(flipped_is_valid.long(), dim=1)
    # 转换回原序列的全局索引
    last_valid_token_global_index = (seq_len - 1) - last_valid_token_in_seq_flipped

    # 2. 获取最后一个有效文档的 ID
    # 使用高级索引获取每个样本最后一个有效 token 的 doc_id
    last_valid_doc_ids_val = batch_doc_ids[torch.arange(batch_size, device=batch_doc_ids.device), last_valid_token_global_index]

    # 3. 找到最后一个有效文档的起始索引
    # 构造一个掩码，标记属于最后一个文档的非padding token
    is_token_in_last_doc_mask = (batch_token_to_doc_map == last_valid_doc_ids_val.unsqueeze(1)) & (batch_token_to_doc_map!= 0)
    # 找到第一个True的索引，即最后一个文档的起始位置
    first_token_in_last_doc_global_index = torch.argmax(is_token_in_last_doc_mask.long(), dim=1)

    # 4. 组装结果 [起始索引, 结束索引]
    batch_last_doc_start_end_indices = torch.stack([
        first_token_in_last_doc_global_index,
        last_valid_token_global_index
    ], dim=1)

    # 5. 处理全 padding 序列的特殊情况
    # 如果整个序列都是 padding，则其最后一个文档的起止索引都应该是 [-1, -1]
    all_padding_mask = (batch_doc_ids == 0).all(dim=1)
    batch_last_doc_start_end_indices = torch.where(
        all_padding_mask.unsqueeze(1).expand(-1, 2), # 扩展为 (batch_size, 2)
        torch.tensor([-1, -1], dtype=torch.long, device=batch_doc_ids.device),
        batch_last_doc_start_end_indices
    )

    return batch_token_to_doc_map, batch_last_token_indices_per_doc, batch_last_doc_start_end_indices

def make_flex_block_causal_mask(
    attention_mask_2d: torch.Tensor,
    query_length=None,
    key_length=None,
    offsets=None,
) -> "BlockMask":
    """
    为 FlexAttention 创建一个块因果文档掩码。
    此函数通过用户定义的 `mask_mod` 函数，生成一个 BlockMask，
    用于高效地表示复杂的注意力模式。
    """
    batch_size, total_seq_len = attention_mask_2d.shape
    if not key_length:
        key_length = total_seq_len
    if not query_length:
        query_length = total_seq_len

    # 预处理文档信息，这些信息将在 mask_mod 中按索引访问
    token_to_doc_map, last_token_indices, batch_last_doc_start_end_indices = prepare_batch_doc_info_optimized(attention_mask_2d)

    device = attention_mask_2d.device

    # Flex Attention 接受一个 mask_mod 函数，该函数定义了注意力计算的逻辑
    def wsa_mask_mod(b, h, q_idx, kv_idx):
        """
        自定义注意力掩码函数。
        b: batch索引
        h: head索引 (在此示例中未使用，因为掩码与头无关)
        q_idx: query token的全局索引
        kv_idx: key/value token的全局索引
        """
        # 获取查询和键的文档ID
        q_doc_id = token_to_doc_map[b, q_idx]
        kv_doc_id = token_to_doc_map[b, kv_idx]

        # 规则1: 文档内部因果注意力
        # 如果查询和键在同一文档内，且键的索引不大于查询的索引（因果关系），且非padding
        rule1_mask = (q_doc_id!= 0) & (q_doc_id == kv_doc_id) & (q_idx >= kv_idx)

        # 获取当前批次项的最后一个文档的起始和结束索引
        # batch_last_doc_start_end_indices 包含 [第一个token的全局索引, 最后一个token的全局索引]
        last_doc_start_id = batch_last_doc_start_end_indices[b][0]
        last_doc_end_id = batch_last_doc_start_end_indices[b][1]

        # 检查 q_idx 是否在最后一个文档内 (包含边界)，并且序列不是全padding
        q_is_in_last_doc = (q_idx >= last_doc_start_id) & (q_idx <= last_doc_end_id) & (last_doc_start_id!= -1)

        # 检查 q_idx 是否是其所属文档的最后一个token (且非padding)
        q_is_last_token_of_its_doc = (last_token_indices[b, q_idx] == q_idx) & (q_doc_id!= 0)

        # 检查 kv_idx 是否是其所属文档的最后一个token (且非padding)
        kv_is_last_token_of_its_doc = (last_token_indices[b, kv_idx] == kv_idx) & (kv_doc_id!= 0)

        # 判断 kv_idx 是否是位于 q_doc_id 之前的某个文档的最后一个token
        kv_is_prev_last_token = (q_doc_id > kv_doc_id) & kv_is_last_token_of_its_doc

        # 规则2: 跨文档注意力 (步幅掩码逻辑)
        # 子规则 2.1: 最后一个文档中的任何token都可以看到前面所有文档的最后一个token
        rule2_part1_mask = q_is_in_last_doc & kv_is_prev_last_token

        # 子规则 2.2: 除最后一个文档外的其他文档的最后一个token，可以看前面所有文档的最后一个token
        rule2_part2_mask = q_is_last_token_of_its_doc & (~q_is_in_last_doc) & kv_is_prev_last_token

        # 结合所有规则：只要满足任一条件，就允许注意力
        final_mask = rule1_mask | rule2_part1_mask | rule2_part2_mask
        return final_mask

    if offsets is not None:
        # 如果有偏移量，则调整索引
        q_offset = offsets
        kv_offset = offsets[1]

        def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
            offset_q = q_idx + q_offset
            offset_kv = kv_idx + kv_offset
            return wsa_mask_mod(batch_idx, head_idx, offset_q, offset_kv)
    else:
        mask_mod = wsa_mask_mod

    # 使用 create_block_mask 创建 BlockMask，并指示进行编译
    return create_block_mask(
        mask_mod=mask_mod,
        B=batch_size,
        H=None,  # H=None 表示 mask_mod 不依赖于 head_idx
        Q_LEN=query_length,
        KV_LEN=key_length,
        device=device,
        _compile=True, # 启用编译以获得更好的性能
    )

# 使用 @torch.compile 装饰器优化 FlexAttention 的前向传播
# 这将把函数编译成优化的计算图，显著减少 Python 开销和操作符融合
@torch.compile
def flex_attention_forward(
    training: bool,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Union,
    scaling: Optional[float] = None,
    **kwargs, # 移除未使用的 softcap 和 head_mask 参数
) -> Tuple:
    block_mask = None
    causal_mask = None
    if isinstance(attention_mask, BlockMask):
        block_mask = attention_mask
    else:
        causal_mask = attention_mask

    if causal_mask is not None:
        # 如果是传统的因果掩码（非BlockMask），则进行切片以匹配键的长度
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]

    enable_gqa = True
    num_local_query_heads = query.shape[1]
    # 如果查询头数不是键头数的整数倍（即存在分组查询注意力 GQA），则重复键值
    if not ((num_local_query_heads & (num_local_query_heads - 1)) == 0): # 检查是否是2的幂，这里逻辑有点问题，应该是检查 num_local_query_heads % key.shape[1] == 0
        # 更准确的 GQA 判断：如果查询头数大于键头数，且能整除
        if num_local_query_heads > key.shape[1] and num_local_query_heads % key.shape[1] == 0:
            key = repeat_kv(key, num_local_query_heads // key.shape[1])
            value = repeat_kv(value, num_local_query_heads // value.shape[1])
            enable_gqa = True # 实际上这里是启用了 GQA 模式下的重复
        else:
            enable_gqa = False # 如果不满足 GQA 条件，则禁用 GQA 优化

    kernel_options = kwargs.get("kernel_options", None)
    attn_output, attention_weights = compile_friendly_flex_attention(
        query,
        key,
        value,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
        scale=scaling,
        kernel_options=kernel_options,
        return_lse=True, # Flex Attention 总是计算 lse，因此始终返回
        training=training,
    )
    # lse (对数和指数) 默认返回 float32，将其转换为与 value 相同的 dtype
    attention_weights = attention_weights.to(value.dtype)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attention_weights

def sdpa_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> Tuple:
    """
    使用 torch.nn.functional.scaled_dot_product_attention (SDPA) 进行注意力计算。
    SDPA 会自动选择最优化实现（如 FlashAttention 或 Memory-Efficient Attention）。
    """
    # 确保 attention_mask 是 4D (B, H, Q, K) 或 3D (B, Q, K)
    if attention_mask.dim() == 3:
        attention_mask = attention_mask.unsqueeze(1).repeat(1, query.shape[1], 1, 1)
    causal_mask = attention_mask
    if attention_mask is not None and causal_mask.ndim == 4:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]

    # 确保输入张量是连续的，以避免某些 PyTorch 版本中 SDPA 的潜在 bug
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # 根据输入形状和掩码判断是否为因果注意力
    if is_causal is None:
        is_causal = query.shape[2] > 1 and causal_mask is None

    # 在 JIT 追踪时，is_causal 可能是 SymBool，需要转换为 bool
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None

if __name__ == "__main__":
    import numpy as np

    # 辅助函数：生成文档ID张量
    def generate_doc_ids(batch_size, sequence_length, min_docs=3, max_docs=None, padding_ratio=0.2):
        """
        生成文档ID张量，满足：
        1. 每个序列至少包含min_docs个文档
        2. 每个文档的长度至少为2个token
        3. 文档ID非严格递增，0表示padding

        参数:
        - batch_size: 批次大小
        - sequence_length: 序列长度
        - min_docs: 每个序列的最小文档数 (默认3)
        - max_docs: 每个序列的最大文档数 (默认根据sequence_length计算)
        - padding_ratio: 填充比例 (0到1之间)

        返回:
        - doc_ids: [batch_size, sequence_length] 的LongTensor
        """
        if max_docs is None:
            max_possible_docs = (sequence_length - int(sequence_length * padding_ratio)) // 2
            max_docs = max(min_docs, max_possible_docs)

        min_actual_length = min_docs * 2
        max_padding = sequence_length - min_actual_length
        if max_padding < 0:
            raise ValueError(f"序列长度({sequence_length})过小，无法满足至少{min_docs}个文档且每个文档长度≥2")

        doc_ids = torch.zeros(batch_size, sequence_length, dtype=torch.long)

        for i in range(batch_size):
            max_actual_length = sequence_length
            actual_length = np.random.randint(min_actual_length, max_actual_length + 1)

            max_possible_docs = actual_length // 2
            num_docs = np.random.randint(min_docs, min(max_docs, max_possible_docs) + 1)

            remaining_tokens = actual_length - (num_docs * 2)
            if remaining_tokens > 0:
                allocations = np.random.multinomial(remaining_tokens, np.ones(num_docs) / num_docs)
            else:
                allocations = np.zeros(num_docs, dtype=int)

            doc_lengths = np.ones(num_docs, dtype=int) * 2 + allocations

            start_idx = 0
            for doc_id in range(1, num_docs + 1):
                length = doc_lengths[doc_id - 1]
                end_idx = start_idx + length
                doc_ids[i, start_idx:end_idx] = doc_id
                start_idx = end_idx

        return doc_ids


    batch_size = 2
    sequence_length = 10 * 1024
    head_dim = 128
    num_heads = 12

    # 确保在CUDA设备上运行以获得最佳性能
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备: {device}")

    # 使用示例 doc_ids，并将其移动到 CUDA 设备
    # doc_ids = torch.LongTensor([[1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    #                             ]).to(device)
    # doc_ids = torch.LongTensor([[1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    #                             [1, 1, 2, 2, 3, 3, 3, 0, 0, 0]]).to(device)
    doc_ids = generate_doc_ids(
        batch_size=batch_size,
        sequence_length=sequence_length,
        min_docs=3,     # 每个序列至少3个文档
        padding_ratio=0.2
    )
    doc_ids = torch.LongTensor(doc_ids).to(device)

    # 生成查询、键、值张量，并将其移动到 CUDA 设备
    query = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device)
    key = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device)
    value = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device)

    # --- Flex Attention (优化后) ---
    print("\n--- Flex Attention (优化后) ---")
    # 为 FlexAttention 创建 BlockMask
    block_mask = make_flex_block_causal_mask(doc_ids)

    # 首次运行 Flex Attention，会触发 torch.compile 编译过程，因此可能较慢
    print("首次运行 Flex Attention (包含编译时间)...")
    t1 = time.time()
    flex_attn_output, attention_weights = flex_attention_forward(True, query, key, value, attention_mask=block_mask)
    t2 = time.time()
    print("======> Flex Attention 输出:")
    print(f"形状: {flex_attn_output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    print(f"Flex Attention (首次运行/编译) 耗时: {t2 - t1:.4f}s")

    # 再次运行 Flex Attention，此时已编译，性能应显著提升
    print("\n再次运行 Flex Attention (已编译)...")
    t1_re_run = time.time()
    flex_attn_output_re_run, _ = flex_attention_forward(True, query, key, value, attention_mask=block_mask)
    t2_re_run = time.time()
    print(f"Flex Attention (已编译，再次运行) 耗时: {t2_re_run - t1_re_run:.4f}s")


    # --- SDPA Attention ---
    print("\n--- SDPA Attention ---")
    # 为 SDPA 生成完整的注意力掩码，并调整维度以匹配 SDPA 期望的 (B, 1, Q, K) 或 (B, H, Q, K)
    attention_mask_sdpa = flex_attention_mask(doc_ids, window_mask=True, stride_mask=True).unsqueeze(1)
    # print(attention_mask_sdpa)

    t3 = time.time()
    sdpa_attn_output, _ = sdpa_attention_forward(query, key=key, value=value, attention_mask=attention_mask_sdpa, is_causal=False)
    t4 = time.time()
    print("======> SDPA Attention 输出:")
    print(f"形状: {sdpa_attn_output.shape}")
    print(f"SDPA Attention 耗时: {t4 - t3:.4f}s")

    print("\n--- 结果对比 ---")
    # 比较 Flex Attention 和 SDPA 的输出是否接近
    print(f"输出是否接近? {torch.allclose(flex_attn_output, sdpa_attn_output, atol=1e-4)}")
    print(f"均方误差 (MSE): {F.mse_loss(flex_attn_output, sdpa_attn_output).item()}")