"""
MongoDB-backed candidate repository for resume ranking system.

This module handles all persistence for the candidate pool in MongoDB.
DEPLOYMENT NOTE: CSV is deprecated. All candidates are stored in MongoDB.

Required fields per candidate:
  - id: unique identifier (UUID string)
  - name: extracted name from resume
  - phone: extracted phone number
  - email: extracted email address
  - predicted_role: tech role classification
  - skills_list: list of extracted skills
  - experience_years_num: numeric years of experience
  - raw_text: full resume text (needed for TF-IDF vectorization at ranking time)
"""

import os
from typing import Dict, List, Optional, Any
from uuid import uuid4

import pandas as pd

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class CandidateRepository:
    """MongoDB-backed candidate pool repository for deployment."""

    def __init__(
        self,
        uri: Optional[str] = None,
        db_name: str = "resume_classifier",
        collection_name: str = "candidates",
    ):
        """
        Initialize MongoDB repository.

        Args:
            uri: MongoDB connection URI. If None, uses MONGO_URI env var 
                 or defaults to 'mongodb://localhost:27017'.
            db_name: Database name (default: resume_classifier).
            collection_name: Collection name (default: candidates).

        Raises:
            ImportError: If pymongo is not installed.
            ConnectionError: If MongoDB connection fails.
        """
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError(
                "pymongo is required for MongoDB backend. "
                "Install it with: pip install pymongo"
            )

        uri = uri or os.getenv("MONGO_URI", "mongodb://localhost:27017")

        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command("ping")
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to MongoDB at {uri}: {e}. "
                "Make sure MongoDB is running."
            )

        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

        # Create index on 'id' for faster lookups and uniqueness
        self.collection.create_index("id", unique=True)

    def load_all(self) -> pd.DataFrame:
        """
        Load all candidates from MongoDB collection into a DataFrame.

        Returns:
            pd.DataFrame: Candidates with all fields needed for ranking.
                         Empty DataFrame with expected columns if no candidates.
        """
        try:
            documents = list(self.collection.find({}, {"_id": 0}))

            if not documents:
                # Return empty DataFrame with expected columns
                return pd.DataFrame(
                    columns=[
                        "id",
                        "name",
                        "phone",
                        "email",
                        "predicted_role",
                        "skills_list",
                        "experience_years_num",
                        "raw_text",
                    ]
                )

            df = pd.DataFrame(documents)

            # Ensure expected columns exist
            expected_cols = [
                "id",
                "name",
                "phone",
                "email",
                "predicted_role",
                "skills_list",
                "experience_years_num",
                "raw_text",
            ]
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = None

            # Ensure skills_list is always a list
            df["skills_list"] = df["skills_list"].apply(
                lambda x: x if isinstance(x, list) else []
            )

            # Ensure raw_text is string
            df["raw_text"] = df["raw_text"].astype(str)

            return df

        except Exception as e:
            raise RuntimeError(f"Failed to load candidates from MongoDB: {e}")

    def add_candidate(self, candidate: Dict[str, Any]) -> str:
        """
        Add or upsert a single candidate in MongoDB.

        Args:
            candidate: Dictionary with fields: id, name, phone, email, predicted_role,
                      skills_list, experience_years_num, raw_text.

        Returns:
            str: The candidate's ID.

        Raises:
            ValueError: If required fields are missing.
            RuntimeError: If MongoDB operation fails.
        """
        # Generate ID if not provided
        if "id" not in candidate or not candidate["id"]:
            candidate["id"] = str(uuid4())

        required_fields = [
            "id",
            "name",
            "phone",
            "email",
            "predicted_role",
            "skills_list",
            "experience_years_num",
            "raw_text",
        ]

        for field in required_fields:
            if field not in candidate:
                raise ValueError(f"Missing required field: {field}")

        try:
            # Upsert: insert if not exists, update if exists
            self.collection.update_one(
                {"id": candidate["id"]}, {"$set": candidate}, upsert=True
            )
            return candidate["id"]
        except Exception as e:
            raise RuntimeError(f"Failed to add candidate to MongoDB: {e}")

    def save_all(self, df: pd.DataFrame) -> None:
        """
        Replace all documents in the collection with DataFrame contents.

        WARNING: This deletes all existing candidates and replaces them.
        Use add_candidate() for incremental updates instead.

        Args:
            df: DataFrame with candidate data.

        Raises:
            RuntimeError: If MongoDB operation fails.
        """
        try:
            # Delete all existing candidates
            self.collection.delete_many({})

            # Insert new candidates
            if len(df) > 0:
                records = df.to_dict("records")
                self.collection.insert_many(records)
        except Exception as e:
            raise RuntimeError(f"Failed to save candidates to MongoDB: {e}")

    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()


def get_candidate_repository() -> CandidateRepository:
    """
    Factory function to get MongoDB repository instance.

    Environment variables:
      - MONGO_URI: MongoDB connection string 
        (default: mongodb://localhost:27017)
      - MONGO_DB: Database name (default: resume_classifier)
      - MONGO_COLLECTION: Collection name (default: candidates)

    Returns:
        CandidateRepository: Configured MongoDB repository instance.

    Raises:
        ConnectionError: If MongoDB connection fails.
        ImportError: If pymongo is not installed.
    """
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    mongo_db = os.getenv("MONGO_DB", "resume_classifier")
    mongo_collection = os.getenv("MONGO_COLLECTION", "candidates")

    return CandidateRepository(
        uri=mongo_uri, db_name=mongo_db, collection_name=mongo_collection
    )
