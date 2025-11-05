#/bin/bash
VLLM_PATH=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
echo "Patching vLLM at $VLLM_PATH"

TARGET_CONNECTOR_FILE="$VLLM_PATH/distributed/kv_transfer/kv_connector/v1/lskp_connector.py"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC_CONNECTOR_FILE="$SCRIPT_DIR/lskp_connector.py"

if [ ! -f "$SRC_CONNECTOR_FILE" ]; then
    echo "Source file not found: $SRC_CONNECTOR_FILE" >&2
    exit 1
fi

cp -f "$SRC_CONNECTOR_FILE" "$TARGET_CONNECTOR_FILE" || { echo "Failed to copy $SRC_CONNECTOR_FILE to $TARGET_CONNECTOR_FILE" >&2; exit 1; }
echo "Copied $SRC_CONNECTOR_FILE -> $TARGET_CONNECTOR_FILE"

TARGET_FACTORY_FILE="$VLLM_PATH/distributed/kv_transfer/kv_connector/factory.py"
SRC_FACTORY_FILE="$SCRIPT_DIR/factory.py"
diff -u $SRC_FACTORY_FILE $TARGET_FACTORY_FILE > diff.patch
patch --dry-run $TARGET_FACTORY_FILE -p0 < diff.patch
if [ $? -eq 0 ]; then
    patch $TARGET_FACTORY_FILE -p0 < diff.patch
    echo "Patched $TARGET_FACTORY_FILE"
else
    echo "$TARGET_FACTORY_FILE is already patched or patching failed." >&2
fi
rm diff.patch


