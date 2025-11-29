import os
import re
import sys

# 获取用户评论内容
user_comment = os.environ.get("USER_COMMENT", "")

# 1. 解析指令
match = re.search(r'/update_page\s+Page:\s*(\S+)\s+Content:\s*(.*)', user_comment, re.DOTALL | re.IGNORECASE)

print("--- Starting Update Script ---")
print(f"Received comment: {user_comment}")

if not match:
    print("Error: Command format incorrect. Use: /update_page Page: <filename> Content: <new_text>")
    sys.exit(1)

file_path = match.group(1).strip()
new_content = match.group(2).strip()

# 2. 写入文件
try:
    full_path = os.
path.join(repo_root, file_path) # <--- 新增的路径构建

    print(f"✅ SUCCESS: Successfully wrote content to {file_path}")

except Exception as e:
    print(f"❌ FAILURE: Could not write to file {file_path}. Error: {e}")
    sys.exit(1)
