"""Entry point script for training the SQLi detection model."""

from sqli_pipeline.trainer import Trainer
from sqli_pipeline.config import Config


def main():
    config = Config()
    trainer = Trainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
