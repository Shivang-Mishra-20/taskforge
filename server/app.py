"""
Task Forge AI — Server Entry Point
Required by openenv-core for multi-mode deployment.
"""
import os
import uvicorn
from app import app


def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()