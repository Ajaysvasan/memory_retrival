import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class SystemVersion:
    """System version tracking"""

    version: str
    dataset_hashes: dict  # filename -> hash mapping
    model_hash: str
    config_hash: str
    timestamp: datetime

    @staticmethod
    def compute_file_hash(filepath: Path) -> str:
        """Compute SHA256 hash of file"""
        sha256 = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except FileNotFoundError:
            return "NOT_FOUND"

    @staticmethod
    def compute_dataset_hashes(dataset_files: List[Path]) -> dict:
        """Compute hashes for all dataset files"""
        hashes = {}
        for filepath in dataset_files:
            hashes[filepath.name] = SystemVersion.compute_file_hash(filepath)
        return hashes

    @staticmethod
    def compute_config_hash() -> str:
        """Compute hash of current configuration"""
        from config import Config

        config_dict = {
            k: str(v)
            for k, v in vars(Config).items()
            if not k.startswith("_") and not callable(v)
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    @classmethod
    def create_current(
        cls, dataset_files: List[Path], model_path: Optional[Path] = None
    ) -> "SystemVersion":
        """Create version for current system state"""

        dataset_hashes = cls.compute_dataset_hashes(dataset_files)
        model_hash = (
            cls.compute_file_hash(model_path)
            if model_path and model_path.exists()
            else "NO_MODEL"
        )
        config_hash = cls.compute_config_hash()

        return cls(
            version="1.0.0",
            dataset_hashes=dataset_hashes,
            model_hash=model_hash,
            config_hash=config_hash,
            timestamp=datetime.now(),
        )

    def is_compatible(self, other: "SystemVersion") -> bool:
        """Check if two versions are compatible"""
        return (
            self.dataset_hashes == other.dataset_hashes
            and self.config_hash == other.config_hash
        )
