import tiktoken


def get_tokenizer(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str) -> int:
    tokenizer = get_tokenizer(model)

    if tokenizer:
        return len(tokenizer.encode(text))
    return estimate_token_count(text)


def estimate_token_count(text: str) -> int:
    return max(1, len(text) // 4)
