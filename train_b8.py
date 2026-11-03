'''
 运行 B8：

  python train_b8.py \
    --config config_v1.yaml \
    --feature-manifest /path/to/RETFound/feature_extraction_manifest.csv \
    --split-manifest /path/to/split_manifest.csv

  运行 B9：

  python train_b9.py \
    --config config_v1.yaml \
    --feature-manifest /path/to/V_JEPA/feature_extraction_manifest.csv \
    --split-manifest /path/to/split_manifest.csv

  输出目录分别为：

  runs/.../b8/
  runs/.../b9/
'''
import sys
from train_b89 import main
if __name__ == "__main__":
    sys.argv[1:1] = ["--task", "b8"]
    main()
