import hashlib
import json
import re
from pathlib import Path
from typing import Optional, List
from datetime import datetime


class LLMCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _sanitize_prefix(self, prefix: str) -> str:
        """Sanitize prefix to be filesystem-safe, handling Unicode characters"""
        # Replace filesystem-problematic characters with underscores
        # Keep Unicode letters, digits, hyphens, and underscores
        sanitized = re.sub(r'[^\w\-_\u0600-\u06FF\u10A0-\u10FF\u1C90-\u1CBF]', '_', prefix, flags=re.UNICODE)
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        # Ensure it's not empty and not too long
        if not sanitized:
            sanitized = "cache"
        return sanitized[:50]  # Limit length
    
    def _get_cache_key(self, prompt: str, model: str) -> str:
        """Generate a unique hash for the prompt+model combination"""
        content = f"{model}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cache_filename(self, prompt: str, model: str, prefix: Optional[str] = None) -> str:
        """Get the cache filename, using prefix + hash if provided, otherwise just hash-based"""
        hash_key = self._get_cache_key(prompt, model)
        if prefix:
            sanitized_prefix = self._sanitize_prefix(prefix)
            return f"{sanitized_prefix}_{hash_key}_llm.json"
        else:
            return f"{hash_key}.json"
    
    def get(self, prompt: str, model: str, prefix: Optional[str] = None) -> Optional[str]:
        try:
            """Retrieve cached response if it exists"""
            cache_file = self.cache_dir / self._get_cache_filename(prompt, model, prefix)
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data['response']
        except Exception as e:
            return None
        
        return None
    
    def set(self, prompt: str, model: str, response: str, prefix: Optional[str] = None):
        """Cache the response"""
        cache_file = self.cache_dir / self._get_cache_filename(prompt, model, prefix)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({
                'prompt': prompt,
                'model': model,
                'response': response,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)

