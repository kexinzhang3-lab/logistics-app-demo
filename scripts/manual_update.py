import os
import re
import sys

# 获取用户评论内容
user_comment = os.environ.get("USER_COMMENT", "")

# 1. 解析指令：寻找 Page: 和 Content: 字段
# 使用正则表达式来提取指令
match = re.search(r'/update_page\s+Page:\s*(\S+)\s+Content:\s*(.*)', user_comment, re.DOTALL | re.IGNORECASE)

if not match:
    print("Error: Command format incorrect. Use: /update_page Page: <filename> Content: <new_text>")
    sys.exit(1)

file_path = match.group(1).strip()
new_content = match.group(2).strip()

if not file_path or not new_content:
    print("Error: File path or content is empty.")
    sys.exit(1)

# 2. 检查文件是否存在并写入内容
try:
    # 假设您的文件在仓库根目录
    full_path = file_path 
    
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ SUCCESS: Successfully wrote content to {full_path}")
    print("The file will be committed to the repository automatically.")

except Exception as e:
    print(f"❌ FAILURE: Could not write to file {full_path}. Error: {e}")
    sys.exit(1)