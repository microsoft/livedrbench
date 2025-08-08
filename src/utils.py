

import hashlib
import base64
from typing import Callable
import os
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import time
from functools import wraps

def derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length key from the password using SHA256.
    Code derived from https://github.com/openai/simple-evals/blob/main/browsecomp_eval.py
    """ 
    
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]

def decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt base64-encoded ciphertext with XOR.
    Code derived from https://github.com/openai/simple-evals/blob/main/browsecomp_eval.py
    """
    
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()

def map_with_progress(
    f: Callable,
    xs: list[dict],
    num_threads: int = os.cpu_count() or 10,
    pbar: bool = True,
):
    """
    Apply f to each element of xs, using a ThreadPool, and show progress.
    """
    pbar_fn = tqdm if pbar else lambda x, *args, **kwargs: x
    
    def apply_kwargs(kwargs):
        return f(**kwargs)

    if os.getenv("debug"):
        return list(map(apply_kwargs, pbar_fn(xs, total=len(xs))))
    else:
        with ThreadPool(min(num_threads, len(xs))) as pool:
            return list(pbar_fn(pool.imap_unordered(apply_kwargs, xs), total=len(xs)))

def retry_on_exception(max_retries=5, delay=0, exceptions=(Exception,)):
    """
    Decorator to retry a function on exception.

    Args:
        max_retries (int): Number of times to retry before giving up.
        delay (float): Delay (in seconds) between retries.
        exceptions (tuple): Exception(s) to catch and retry on.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt <= max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        print(f"Failed to execute {func.__name__} with args {args} and kwargs {kwargs} after {max_retries} attempts. Last exception: {e}")
                        raise
                    attempt += 1
                    if delay:
                        time.sleep(delay)
        return wrapper
    return decorator
