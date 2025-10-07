import hashlib
import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime


class LLMCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, prompt: str, model: str) -> str:
        """Generate a unique hash for the prompt+model combination"""
        content = f"{model}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, prompt: str, model: str) -> Optional[str]:
        try:
            """Retrieve cached response if it exists"""
            cache_file = self.cache_dir / f"{self._get_cache_key(prompt, model)}.json"
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data['response']
        except Exception as e:
            return None
        
        return None
    
    def set(self, prompt: str, model: str, response: str):
        """Cache the response"""
        cache_file = self.cache_dir / f"{self._get_cache_key(prompt, model)}.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({
                'prompt': prompt,
                'model': model,
                'response': response,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)

