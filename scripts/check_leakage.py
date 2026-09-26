import re
import hashlib

'''
Xoá khoảng trắng thừa trong code
int add(int a,   int b) {    return a + b;     }
int add(int a, int b) { return a + b; }
'''
def normalize_code(code):
    return re.sub(r"\s+", " ", str(code)).strip()
'''
Dùng để phát hiện hàm trùng lặp / rò rỉ dữ liệu giữa train và test
int add(int a,   int b) {    return a + b;     }  -> 'a1b2c3...'
int add(int a, int b) { return a + b; }           -> 'a1b2c3...' 
'''
def code_fingerprint(code):
    return hashlib.sha256(normalize_code(code).encode("utf-8")).hexdigest()

