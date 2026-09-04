# Add to the top of dataset.py
import hashlib
import pickle
from functools import lru_cache

# ============================================================
# AUDIO CACHE - SPEEDS UP LOADING DRAMATICALLY
# ============================================================

class AudioCache:
    """Cache loaded audio files to avoid repeated disk reads"""
    
    def __init__(self, max_size=5000, cache_dir="audio_cache"):
        self.cache = {}
        self.max_size = max_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.hits = 0
        self.misses = 0
    
    def get(self, path):
        """Get audio from cache or load it"""
        path = str(Path(path).resolve())
        
        # Check memory cache
        if path in self.cache:
            self.hits += 1
            return self.cache[path]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{hashlib.md5(path.encode()).hexdigest()}.pkl"
        if cache_file.exists():
            try:
                audio = pickle.load(open(cache_file, 'rb'))
                if len(self.cache) < self.max_size:
                    self.cache[path] = audio
                self.hits += 1
                return audio
            except:
                pass
        
        # Load from disk
        audio = load_audio(path)
        
        # Save to disk cache
        try:
            pickle.dump(audio, open(cache_file, 'wb'))
        except:
            pass
        
        # Save to memory cache
        if len(self.cache) < self.max_size:
            self.cache[path] = audio
        
        self.misses += 1
        return audio
    
    def stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return f"Cache hit rate: {hit_rate:.1%} ({self.hits}/{total})"

# Global cache instance
_audio_cache = AudioCache(max_size=5000)

def load_audio_cached_fast(path):
    """Fast cached audio loader"""
    return _audio_cache.get(path)

# Replace the existing load_audio function to use cache
def load_audio(path):
    return load_audio_cached_fast(path)