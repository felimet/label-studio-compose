"""WSGI entry point for SAM2.1 image backend."""
import os
import argparse
import json
import logging
import logging.config

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": os.getenv("LOG_LEVEL", "INFO"),
            "stream": "ext://sys.stdout",
            "formatter": "standard",
        }
    },
    "root": {
        "level": os.getenv("LOG_LEVEL", "INFO"),
        "handlers": ["console"],
        "propagate": True,
    },
})

from label_studio_ml.api import init_app  # noqa: E402
from model import NewModel               # noqa: E402

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def _get_kwargs_from_config(config_path: str = _DEFAULT_CONFIG_PATH) -> dict:
    if not os.path.exists(config_path):
        return {}
    with open(config_path) as f:
        config = json.load(f)
    assert isinstance(config, dict)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2.1 Image ML Backend")
    parser.add_argument("-p", "--port", dest="port", type=int, default=9090)
    parser.add_argument("--host", dest="host", type=str, default="0.0.0.0")
    parser.add_argument("-d", "--debug", dest="debug", action="store_true")
    parser.add_argument(
        "--log-level", dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None,
    )
    parser.add_argument("--basic-auth-user", default=os.environ.get("BASIC_AUTH_USER"))
    parser.add_argument("--basic-auth-pass", default=os.environ.get("BASIC_AUTH_PASS"))
    args = parser.parse_args()

    if args.log_level:
        logging.root.setLevel(args.log_level)

    app = init_app(
        model_class=NewModel,
        basic_auth_user=args.basic_auth_user,
        basic_auth_pass=args.basic_auth_pass,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)

else:
    # gunicorn / uWSGI entry point
    app = init_app(model_class=NewModel)
