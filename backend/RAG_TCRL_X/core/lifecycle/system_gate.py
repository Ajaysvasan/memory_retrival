import pickle
from pathlib import Path
from typing import Optional

from config import Config
from core.lifecycle.versioning import SystemVersion
from logger import Logger


class SystemGate:
    """Controls system initialization and version compatibility"""

    def __init__(self, config):
        self.config = config
        self.logger = Logger().get_logger("SystemGate")
        self.version_file = config.DATA_DIR / "system_version.pkl"

    def check_initialization_required(self) -> bool:
        """Check if full initialization is required"""
        required_files = [
            self.config.EMBEDDINGS_PATH,
            self.config.TOPICS_PATH,
            self.config.FAISS_DIR,
        ]

        # Check if all required files exist
        files_exist = all(Path(f).exists() for f in required_files)

        if not files_exist:
            self.logger.info("Initialization required: Missing required files")
            return True

        # Check version compatibility
        if not self.version_file.exists():
            self.logger.info("Initialization required: No version file found")
            return True

        try:
            saved_version = self._load_version()
            dataset_files = Config.get_dataset_files()
            current_version = SystemVersion.create_current(dataset_files)

            if not current_version.is_compatible(saved_version):
                self.logger.warning("Initialization required: Version mismatch")

                # Show what changed
                added_files = set(current_version.dataset_hashes.keys()) - set(
                    saved_version.dataset_hashes.keys()
                )
                removed_files = set(saved_version.dataset_hashes.keys()) - set(
                    current_version.dataset_hashes.keys()
                )
                modified_files = {
                    f
                    for f in current_version.dataset_hashes.keys()
                    & saved_version.dataset_hashes.keys()
                    if current_version.dataset_hashes[f]
                    != saved_version.dataset_hashes[f]
                }

                if added_files:
                    self.logger.warning(f"Added files: {', '.join(added_files)}")
                if removed_files:
                    self.logger.warning(f"Removed files: {', '.join(removed_files)}")
                if modified_files:
                    self.logger.warning(f"Modified files: {', '.join(modified_files)}")

                return True

            self.logger.info("Using existing initialization (versions compatible)")
            return False

        except Exception as e:
            self.logger.error(f"Version check failed: {e}")
            return True

    def save_version(self):
        """Save current system version"""
        dataset_files = Config.get_dataset_files()
        version = SystemVersion.create_current(
            dataset_files,
            Config.RL_MODEL_PATH if Config.RL_MODEL_PATH.exists() else None,
        )

        with open(self.version_file, "wb") as f:
            pickle.dump(version, f)

        self.logger.info(f"Saved system version: {version.version}")
        self.logger.info(f"Dataset files tracked: {len(version.dataset_hashes)}")

    def _load_version(self) -> SystemVersion:
        """Load saved system version"""
        with open(self.version_file, "rb") as f:
            return pickle.load(f)

    def validate_runtime_requirements(self):
        """Validate runtime requirements"""
        # Check dataset directory
        if not self.config.DATASET_DIR.exists():
            raise RuntimeError(
                f"Dataset directory not found: {self.config.DATASET_DIR}"
            )

        # Check for dataset files
        dataset_files = Config.get_dataset_files()
        if not dataset_files:
            raise RuntimeError(
                f"No dataset files found in {self.config.DATASET_DIR}\n"
                f"Supported formats: {', '.join(Config.SUPPORTED_FORMATS)}\n"
                f"Please add at least one dataset file."
            )

        self.logger.info(f"Found {len(dataset_files)} dataset files")
        for filepath in dataset_files:
            self.logger.info(f"  - {filepath.name} ({filepath.suffix})")

        self.logger.info("Runtime requirements validated")
