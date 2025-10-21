import yaml,traceback,sys
from torchdistill.models.util import redesign_model
cfg = yaml.safe_load(open("configs/sample/cifar10/kd/resnet20_from_densenet_bc_k12_depth100-final_run.yaml"))
mc = cfg["models"]["student_model"]
print("model_config:", mc)
try:
    m = redesign_model(None, mc, mc.get("key"))
    print("BUILT model type:", type(m))
except Exception:
    traceback.print_exc()
    sys.exit(1)
