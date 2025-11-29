import os
import re
import sys

# 获取用户评论内容
user_comment = os.environ.get("USER_COMMENT", "")
repo_root = os.getcwd() # <--- 请加入这行代码！

# 1. 解析指令
match = re.search(r'/update_page\s+Page:\s*(\S+)\s+Content:\s*(.*)', user_comment, re.DOTALL | re.IGNORECASE)
# ... 中间代码不变 ...
# ...
file_path = match.group(1).strip()
new_content = match.group(2).strip()

# 2. 写入文件
try:
    full_path = os.path.join(repo_root, file_path) # <--- 请加入这行代码！

    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    # ... 底部代码不变 ...
