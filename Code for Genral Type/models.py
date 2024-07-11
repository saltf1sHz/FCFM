import copy
import math
import torch
from torch import nn
import numpy as np

def get_positional_encoding(max_seq_len, hidden_size):
    positional_encoding = np.array([
        [pos / np.power(10000, 2 * i / hidden_size) for i in range(hidden_size)]
        if pos != 0 else np.zeros(hidden_size) for pos in range(max_seq_len)])
    positional_encoding[1:, 0::2] = np.sin(positional_encoding[1:, 0::2])
    positional_encoding[1:, 1::2] = np.cos(positional_encoding[1:, 1::2])
    return torch.from_numpy(positional_encoding)

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu}


class BertSelfAttention(nn.Module):
    """自注意力机制层, 见Transformer(一), 讲编码器(encoder)的第2部分"""
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        # 判断embedding dimension是否可以被num_attention_heads整除
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Q, K, V线性映射
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # 输入x为QKV中的一个, 维度: [batch_size, seq_length, embedding_dim]
        # 输出的维度经过reshape和转置: [batch_size, num_heads, seq_length, embedding_dim / num_heads]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, get_attention_matrices=False):
        # Q, K, V线性映射
        # Q, K, V的维度为[batch_size, seq_length, num_heads * embedding_dim]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        # 把QKV分割成num_heads份
        # 把维度转换为[batch_size, num_heads, seq_length, embedding_dim / num_heads]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # Q与K求点积
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # attention_scores: [batch_size, num_heads, seq_length, seq_length]
        # 除以K的dimension, 开平方根以归一为标准正态分布
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask
        # attention_mask 注意力矩阵mask: [batch_size, 1, 1, seq_length]
        # 元素相加后, 会广播到维度: [batch_size, num_heads, seq_length, seq_length]

        # softmax归一化, 得到注意力矩阵
        # Normalize the attention scores to probabilities.
        attention_probs_ = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs_)

        # 用注意力矩阵加权V
        context_layer = torch.matmul(attention_probs, value_layer)
        # 把加权后的V reshape, 得到[batch_size, length, embedding_dimension]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # 输出attention矩阵用来可视化
        if get_attention_matrices:
            return context_layer, attention_probs_
        return context_layer, None

class BertLayerNorm(nn.Module):
    """LayerNorm层, 见Transformer(一), 讲编码器(encoder)的第3部分"""
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertSelfOutput(nn.Module):
    # 封装的LayerNorm和残差连接, 用于处理SelfAttention的输出
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    # 封装的多头注意力机制部分, 包括LayerNorm和残差连接
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, get_attention_matrices=False):
        self_output, attention_matrices = self.self(input_tensor, attention_mask, get_attention_matrices=get_attention_matrices)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_matrices


class BertIntermediate(nn.Module):
    # 封装的FeedForward层和激活层
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    # 封装的LayerNorm和残差连接, 用于处理FeedForward层的输出
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    # 一个transformer block
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, get_attention_matrices=False):
        # Attention层(包括LayerNorm和残差连接)
        attention_output, attention_matrices = self.attention(hidden_states, attention_mask, get_attention_matrices=get_attention_matrices)
        # FeedForward层
        intermediate_output = self.intermediate(attention_output)
        # LayerNorm与残差连接输出层
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_matrices
    

class BertEncoder(nn.Module):
    # transformer blocks * N
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        # 复制N个transformer block
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, get_attention_matrices=False):
        """
        :param output_all_encoded_layers: 是否输出每一个transformer block的隐藏层计算结果
        :param get_attention_matrices: 是否输出注意力矩阵, 可用于可视化
        """
        all_attention_matrices = []
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states, attention_matrices = layer_module(hidden_states, attention_mask, get_attention_matrices=get_attention_matrices)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                all_attention_matrices.append(attention_matrices)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            all_attention_matrices.append(attention_matrices)
        return all_encoder_layers, all_attention_matrices
    
class BertPooler(nn.Module):
    """Pooler是把隐藏层(hidden state)中对应#CLS#的token的一条提取出来的功能"""
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

""" class CrazyThursdayPredictionHead(nn.Module):
    def __init__(self, config):
        super(CrazyThursdayPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.output_size)
        # self.scalar = torch.FloatTensor([10, 0.002, 1, 1, 1e-8, 1]).cuda()
        self.scalar = torch.FloatTensor([10, 1, 1, 1, 1e-3, 1]).cuda()
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        y = self.decoder(hidden_states)
        y[:,1:] = torch.sigmoid(y[:,1:])
        return y """



class CrazyThursdayPredictionHead(nn.Module):
    def __init__(self, config):
        super(CrazyThursdayPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.output_size)
        # self.scalar = torch.FloatTensor([10, 0.002, 1, 1, 1e-8, 1]).cuda()
        # self.scalar = torch.FloatTensor([1, 1e-3, 1, 1e-7, 1, 1]).cuda()
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        y = self.decoder(hidden_states)
        # y[:, 0] = 1.5 * torch.sigmoid(y[:, 0])
        y[:, 1:] = torch.sigmoid(y[:, 1:])
        return y 



def init_positional_encoding(hidden_dim, max_seq_len):
    position_enc = np.array([
        [pos / np.power(10000, 2 * i / hidden_dim) for i in range(hidden_dim)]
        if pos != 0 else np.zeros(hidden_dim) for pos in range(max_seq_len)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    denominator = np.sqrt(np.sum(position_enc**2, axis=1, keepdims=True))
    position_enc = position_enc / (denominator + 1e-8)
    position_enc = torch.from_numpy(position_enc).type(torch.FloatTensor)

    return position_enc

class CrazyThursdayEmbeddings(nn.Module):
    def __init__(self, config):
        super(CrazyThursdayEmbeddings, self).__init__()
        self.config = config
        self.mapping_layer = nn.Linear(config.input_size, config.hidden_size)
        positional_enc = init_positional_encoding(self.config.hidden_size, self.config.max_position_embeddings)
        self.positional_enc = torch.unsqueeze(positional_enc, dim=0).cuda()
    def forward(self, input_tensor):
        output_tensor = self.mapping_layer(input_tensor)
        positional_enc = self.positional_enc[:, :input_tensor.size()[1], :]
        return output_tensor + positional_enc




class CrazyThursdayKFC(nn.Module):
    def __init__(self, config):
        super(CrazyThursdayKFC, self).__init__()
        self.embeddings = CrazyThursdayEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.decoder = CrazyThursdayPredictionHead(config)

    def forward(self, x, attention_mask, output_all_encoded_layers=False, get_attention_matrices=False):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        embedding_output = self.embeddings(x)
        encoded_layers, all_attention_matrices = self.encoder(embedding_output,
                                                            extended_attention_mask,
                                                            output_all_encoded_layers=output_all_encoded_layers,
                                                            get_attention_matrices=get_attention_matrices)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        y = self.decoder(pooled_output)
        return y