# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.


import re
import requests
from urllib.parse import quote
from typing import Optional, NamedTuple


class ModelDownloadInfo(NamedTuple):
    model_id: str
    newer_version_id: Optional[str]
    web_url: str
    artifact: Optional["ModelArtifact"]


class ModelArtifact(NamedTuple):
    download_url: str
    view_url: str
    host_name: str
    doi: str


class ModelNotFoundError(Exception):
    """Raised when a model is not found in the repository"""

    pass


class NequIPNetAPIClient:
    MODEL_ID_PATTERN = re.compile(r"^([a-zA-Z0-9-]+)/([a-zA-Z0-9-]+):([a-zA-Z0-9-.]+)$")
    BASE_URL = "https://www.nequip.net/"

    def __init__(
        self,
        user_agent: Optional[str] = None,
    ):
        """
        Initialize the API client

        Args:
            user_agent: Custom User-Agent string (optional)
        """
        self.session = None
        self.user_agent = user_agent or "NequipAPIClient/1.0 (Python)"

    def get_model_download_info(self, model_id: str) -> ModelDownloadInfo:
        # Validate model_id format
        if not self.MODEL_ID_PATTERN.match(model_id):
            raise ValueError(
                f"Invalid model_id format: {model_id}. "
                "Expected format: namespace/model:version (e.g., 'my-org/my-model:v1')"
            )

        # URL encode the model ID
        encoded_id = quote(model_id, safe="")

        url = f"{self.BASE_URL}/api/models/download/{encoded_id}"
        response = self.session.get(url)

        if response.status_code == 404:
            raise ModelNotFoundError(f"Model not found on nequip.net: {model_id}")

        response.raise_for_status()

        # Parse the response
        data = response.json()

        # Create ModelArtifact if artifact data exists
        artifact = None
        if data.get("artifact"):
            artifact_data = data["artifact"]
            artifact = ModelArtifact(
                download_url=artifact_data.get("downloadUrl"),
                view_url=artifact_data.get("viewUrl"),
                host_name=artifact_data.get("hostName"),
                doi=artifact_data.get("doi", ""),
            )
        assert artifact is not None, (
            f"No artifact found for model {model_id};  please file an issue."
        )

        # Create and return the named tuple
        return ModelDownloadInfo(
            model_id=data.get("modelId"),
            newer_version_id=data.get("newerVersionId"),
            web_url=data.get("webUrl", ""),
            artifact=artifact,
        )

    def __enter__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
