"""Entry point script for evaluating the SQLi detection model."""

from sqli_pipeline.evaluator import Evaluator
from sqli_pipeline.config import Config


def main():
    config = Config()
    evaluator = Evaluator(config)
    evaluator.run()


if __name__ == "__main__":
    main()
