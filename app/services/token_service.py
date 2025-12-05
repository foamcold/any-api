import tiktoken
from typing import List, Dict, Any

class TokenService:
    """
    一个专门用于计算Token的服务。
    它缓存分词器以提高性能，并提供方法来计算输入和输出的Token。
    """
    _cached_encoders = {}

    def _get_tokenizer(self, model_name: str = "gpt-3.5-turbo"):
        """
        根据模型名称获取并缓存tiktoken编码器。
        如果找不到特定模型的编码器，则回退到 'cl100k_base'。
        """
        # 为模型名称提供一个默认值，以防它是空的
        if not model_name:
            model_name = "gpt-3.5-turbo"
            
        if model_name not in self._cached_encoders:
            try:
                self._cached_encoders[model_name] = tiktoken.encoding_for_model(model_name)
            except KeyError:
                # 对于未知模型，使用一个通用的基础编码器
                self._cached_encoders[model_name] = tiktoken.get_encoding("cl100k_base")
        return self._cached_encoders[model_name]

    def count_input_tokens(self, messages: List[Dict[str, Any]], model_name: str) -> int:
        """
        根据OpenAI的规则，为消息列表计算输入Token数。
        参考: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """
        tokenizer = self._get_tokenizer(model_name)
        
        # 针对不同模型的Token计算规则
        if "gpt-3.5-turbo" in model_name:
            tokens_per_message = 4  # 每个消息有4个Token的开销 (role, content等)
            tokens_per_name = -1  # 如果有name字段，则有-1的Token折扣
        elif "gpt-4" in model_name:
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            # 对于未知或通用模型，使用一个合理的默认值
            tokens_per_message = 3
            tokens_per_name = 1

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if value:
                    # 确保所有值都为字符串
                    num_tokens += len(tokenizer.encode(str(value)))
                if key == "name":
                    num_tokens += tokens_per_name
        
        num_tokens += 3  # 每个回复都以 <|start|>assistant<|message|> 作为引导
        return num_tokens

    def count_output_tokens(self, text: str, model_name: str) -> int:
        """
        计算给定文本字符串的Token数。
        """
        if not text:
            return 0
        tokenizer = self._get_tokenizer(model_name)
        return len(tokenizer.encode(text))

# 创建一个单例，方便在其他服务中导入和使用
token_service = TokenService()