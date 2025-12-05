def _merge_messages(messages: list) -> list:
    """
    合并连续的、角色相同的消息。
    - 相邻的 user 消息使用 --- 分隔符。
    - 其他相邻的同角色消息使用换行符合并。
    """
    if not messages:
        return []

    # 使用 .copy() 避免修改原始列表中的字典
    merged = [messages[0].copy()]
    for i in range(1, len(messages)):
        current_msg = messages[i]
        last_msg = merged[-1]

        # 使用 --- 作为分隔符合并相邻的 user 消息
        if current_msg.get("role") == "user" and last_msg.get("role") == "user":
            if "content" in last_msg and isinstance(last_msg["content"], str):
                # 使用高关注度的分隔符
                last_msg["content"] += "\n\n---\n\n" + current_msg.get("content", "")
        # 合并其他同角色消息
        elif current_msg.get("role") == last_msg.get("role"):
             if "content" in last_msg and isinstance(last_msg["content"], str):
                last_msg["content"] += "\n" + current_msg.get("content", "")
        else:
            merged.append(current_msg.copy())
    
    return merged