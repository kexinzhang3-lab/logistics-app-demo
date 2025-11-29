import os
import re
import sys

# 获取用户评论内容
user_comment = os.environ.get("USER_COMMENT", "")
# 仓库根目录路径 (Action 运行环境)
repo_root = os.getcwd() 

# 1. 解析指令
match = re.search(r'/update_page\s+Page:\s*(\S+)\s+Content:\s*(.*)', user_comment, re.DOTALL | re.IGNORECASE)

print("--- Starting Update Script ---")
print(f"Received comment: {user_comment}")

if not match:
    print("Error: Command format incorrect. Use: /update_page Page: <filename> Content: <new_text>")
    sys.exit(1)

file_path = match.group(1).strip()
new_content = match.group(2).strip()

# 2. 检查并写入文件（使用绝对路径确保成功）
try:
    full_path = os.path.join(repo_root, file_path) # 构建绝对路径

    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ SUCCESS: Successfully wrote content to {full_path}")

except Exception as e:
    # 如果文件写入失败，则会报错
    print(f"❌ FAILURE: Could not write to file {full_path}. Error: {e}")
    sys.exit(1)


