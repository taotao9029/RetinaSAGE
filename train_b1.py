'''
 \python train_b1.py
    --config config_v1.yaml \
    --feature-manifest /path/to/feature_extraction_manifest.csv \
    --split-manifest /path/to/split_manifest.csv

  其他实验只需替换启动文件：

  python train_b2.py --config config_v1.yaml
  python train_b3.py --config config_v1.yaml
  python train_b4.py --config config_v1.yaml
  python train_b5.py --config config_v1.yaml
'''
import sys
from train_b12345 import main
if __name__ == "__main__":
    sys.argv[1:1] = ["--task", "b1"]
    main()
