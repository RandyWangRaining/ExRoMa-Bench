# Publishing ExRoMa Assets

The canonical asset bundle is the ModelScope dataset
[`ruilin.wang/ExRoMa-Assets`](https://modelscope.ai/datasets/ruilin.wang/ExRoMa-Assets).
The upload uses ModelScope Hub's resumable HTTP uploader, so an interrupted
upload can continue without restarting completed files.

## 1. Prepare an access token

Create an Access Token in the ModelScope account settings. Authenticate through
the interactive prompt so the token is not placed in shell history:

```bash
conda activate openspace
modelscope-hub --endpoint https://www.modelscope.ai login
modelscope-hub --endpoint https://www.modelscope.ai whoami
```

The login command persists the resulting credentials under
`~/.modelscope/credentials/`. Alternatively, expose
`MODELSCOPE_API_TOKEN` only to the current process environment. Never write the
token into this repository, `.env`, or a command-line argument.

## 2. Install the upload client

The normal ExRoMa compatibility requirements include `modelscope-hub`:

```bash
cd /path/to/ExRoMa
conda activate exroma
python -m pip install -r requirements/isaacsim-compatible.txt
```

## 3. Upload or resume the asset bundle

```bash
cd /path/to/ExRoMa
python scripts/publish_assets_modelscope.py \
  /path/to/ExRoMa-assets \
  --repo-id ruilin.wang/ExRoMa-Assets
```

The uploader writes an ignored `.ms_upload_cache` progress file inside the
asset folder. Run the same command again after a network interruption; files
already committed to ModelScope are skipped.

If the remote repository is missing files that the cache already marks as
committed, bypass the cache and upload only those paths:

```bash
python scripts/publish_assets_modelscope.py \
  /path/to/ExRoMa-assets \
  --force \
  --allow-pattern path/to/missing_asset.usdz
```

After the first upload, set the dataset visibility to **Public** on its
ModelScope settings page.

## 4. Verify the public download path

Use a new temporary directory so the check cannot pass through the existing
local asset link:

```bash
cd /path/to/ExRoMa
exroma assets pull --root /tmp/exroma-assets-modelscope-check
EXROMA_ASSET_ROOT=/tmp/exroma-assets-modelscope-check exroma assets verify
```

If an environment variable was used instead of the login command, remove it
from the current shell after verification:

```bash
unset MODELSCOPE_API_TOKEN
unset MODELSCOPE_ENDPOINT
```
