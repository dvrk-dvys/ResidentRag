import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class Hit:
    #'Hit' as in the hybrid search has found some options
    id: str
    title: str
    text: str
    rrf_score: float = 0.0
    source_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def in_docker():
    return os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER") == "1"
