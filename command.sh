cd /home/user/chensihang/MobileMemBench

PYTHONPATH=. MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    conda run -n robocasa python scripts/eval/run_remote.py \
    --endpoint ws://114.214.164.121:8000 \
    --suite configs/eval/suites/wx01_v1.yaml \
    --output-root outputs/eval
