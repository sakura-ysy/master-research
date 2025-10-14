import hashlib
import base64
import re

def _normalize(text: str) -> bytes:
    # 规范化：统一换行、去 BOM、去尾随空白；按需调整
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()
    return text.encode("utf-8", "surrogatepass")

def hash_prompt_sha256(
    prompt: str,
    *, 
    salt: str | None = None,
    bits: int = 128,          # 输出位长：128/160/192/224/256…
    encoding: str = "base62"  # "hex" | "base64" | "base62"
) -> str:
    """
    生成稳定的哈希 ID：
    - 使用 SHA-256
    - 可选 salt 做命名空间隔离
    - 支持 hex/base64/base62 编码
    - 可指定截断位长（默认 128 bit 足够大多数去重/索引）
    """
    assert 1 <= bits <= 256 and bits % 8 == 0
    h = hashlib.sha256()
    if salt:
        h.update(b"\x00ns:" + salt.encode("utf-8") + b"\x00")
    h.update(_normalize(prompt))
    digest = h.digest()[: bits // 8]

    if encoding == "hex":
        return digest.hex()
    elif encoding == "base64":
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    elif encoding == "base62":
        # 简单 base62，无依赖
        alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        n = int.from_bytes(digest, "big")
        if n == 0:
            return alphabet[0]
        out = []
        while n > 0:
            n, r = divmod(n, 62)
            out.append(alphabet[r])
        return "".join(reversed(out))
    else:
        raise ValueError("encoding must be one of: hex/base64/base62")

# # 示例
# if __name__ == "__main__":
#     p = "You can hash any length of prompt here.\n换行、空格都会影响结果。"
#     print("HEX-128:", hash_prompt_sha256(p, bits=128, encoding="hex"))
#     print("B62-128:", hash_prompt_sha256(p, bits=128, encoding="base62"))
#     print("B64-256:", hash_prompt_sha256(p, bits=256, encoding="base64"))
