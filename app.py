import logging

import uvicorn

from app.config import get_settings


def main() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    settings = get_settings()
    uvicorn.run(
        "app.server:app",
        host=settings.app_host,
        port=settings.app_port,
        log_level="debug",
        reload=False,
    )


if __name__ == "__main__":
    main()
