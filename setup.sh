set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"

pip install -r "${ROOT_DIR}/requirements.txt"
pip install pydantic-settings
sudo apt update
sudo apt-get install iproute2 redis -y