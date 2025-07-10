import argparse
import logging

from arc_tigers.data.config import HFDataConfig
from arc_tigers.data.generate import generate_splits

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset")
    parser.add_argument("data_config", type=str, help="Path to the data config file")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of the full splits even if they already exist.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = HFDataConfig.from_path(args.data_config)
    generate_splits(config, args.force)
