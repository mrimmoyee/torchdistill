import os
import traceback
import inspect
import pprint
import torch
from torchdistill.common.yaml_util import load_yaml_file
from torchdistill.core.training import get_training_box
from torchdistill.models.util import redesign_model
from torchdistill import models as models_pkg
import torchdistill.core.interfaces.registry as registry

def pick_forward_proc_key():
    regs = {name: list(val.keys()) for name, val in registry.__dict__.items() if isinstance(val, dict)}
    for candidate in ("PRE_FORWARD_PROC_FUNC_DICT", "FORWARD_PROC_FUNC_DICT"):
        if candidate in regs and regs[candidate]:
            if "default_pre_forward_process" in regs[candidate]:
                return "default_pre_forward_process"
            return regs[candidate][0]
    return "default_pre_forward_process"

def build_model_from_config(model_config):
    """
    Build a concrete org_model (e.g. resnet20) then call:
      redesign_model(org_model, model_config, model_label)
    """
    if model_config is None:
        return None

    key = model_config.get("key")
    kwargs = model_config.get("kwargs", {}) or {}

    base_model = None
    cls_mod = getattr(models_pkg, "classification", None)
    if cls_mod is not None:
        # resnet
        resnet_mod = getattr(cls_mod, "resnet", None)
        if resnet_mod is not None:
            try:
                if hasattr(resnet_mod, key):
                    base_model = getattr(resnet_mod, key)(**kwargs)
                else:
                    # fallback: parse names like "resnet20" -> resnet(depth, num_classes, pretrained)
                    import re
                    m = re.match(r"resnet(\d+)$", str(key))
                    if m and hasattr(resnet_mod, "resnet"):
                        depth = int(m.group(1))
                        num_classes = kwargs.get("num_classes", 10)
                        pretrained = kwargs.get("pretrained", False)
                        base_model = getattr(resnet_mod, "resnet")(depth, num_classes, pretrained, progress=False, **{k: v for k, v in kwargs.items() if k not in ("num_classes", "pretrained")})
            except Exception:
                base_model = None
        # densenet / wide_resnet fallback
        if base_model is None:
            for sub in ("densenet", "wide_resnet"):
                submod = getattr(cls_mod, sub, None)
                if submod is None:
                    continue
                try:
                    if hasattr(submod, key):
                        base_model = getattr(submod, key)(**kwargs)
                        break
                except Exception:
                    base_model = None

    # MODEL_DICT fallback
    if base_model is None:
        model_dict = getattr(models_pkg, "MODEL_DICT", None)
        if isinstance(model_dict, dict) and key in model_dict:
            try:
                base_model = model_dict[key](**kwargs)
            except Exception:
                base_model = None

    # call redesign_model with a real org_model (if available)
    model = redesign_model(base_model, model_config, key)
    if model is None:
        raise RuntimeError(f"redesign_model returned None for key={key}; ensure base model construction succeeded")
    return model

def main():
    cfg_path = "configs/sample/cifar10/kd/resnet20_from_densenet_bc_k12_depth100-final_run.yaml"
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(cfg_path)
    cfg = load_yaml_file(cfg_path)

    dataset_dict = cfg.get("datasets") or cfg.get("dataset_dict")
    train_config = cfg.get("train") or cfg.get("train_config") or cfg.get("training")
    if dataset_dict is None or train_config is None:
        raise RuntimeError("Config missing 'datasets' or 'train' sections")

    # Ensure train_config has a model entry TrainingBox expects
    if isinstance(train_config, dict) and "model" not in train_config:
        models = cfg.get("models")
        if isinstance(models, dict) and models:
            student = models.get("student_model") or next(iter(models.values()))
            if isinstance(student, dict):
                train_config["model"] = dict(student)

    # ensure forward_proc set to a valid registry key
    fp_key = pick_forward_proc_key()
    if isinstance(train_config.get("model"), dict):
        train_config["model"].setdefault("forward_proc", fp_key)

    # Build model from model_config and pass it to TrainingBox
    model_config = train_config.get("model") if isinstance(train_config, dict) else None
    model = build_model_from_config(model_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    distributed = False
    lr_factor = 1.0

    pprint.pprint({"cfg_path": cfg_path, "device": str(device), "dataset_keys": list(dataset_dict.keys()), "model_type": type(model).__name__})
    tb = get_training_box(model, dataset_dict, train_config, device, device_ids, distributed, lr_factor, accelerator=None)
    # Simple single-stage training loop using TrainingBox APIs
    for epoch in range(tb.num_epochs):
        # pre-epoch hook
        try:
            tb.pre_epoch_process(epoch)
        except Exception:
            pass

        if tb.train_data_loader is None:
            print("No train_data_loader available; aborting training")
            break

        tb.model.train()
        for step, sample_batch in enumerate(tb.train_data_loader):
            # pre-forward hook (may prepare targets/supp_dict)
            try:
                tb.pre_forward_process(sample_batch)
            except Exception:
                pass

            loss = tb.forward_process(sample_batch)

            # backward + optimizer step (supports grad accumulation)
            if tb.optimizer is not None:
                loss.backward()
                accum = getattr(tb, "grad_accum_step", 1) or 1
                if (step + 1) % accum == 0:
                    if getattr(tb, "max_grad_norm", None):
                        torch.nn.utils.clip_grad_norm_(tb.model.parameters(), tb.max_grad_norm)
                    tb.optimizer.step()
                    tb.optimizer.zero_grad()

            # post-forward hook (logging / metrics)
            try:
                tb.post_forward_process(step)
            except Exception:
                pass

        # post-epoch hook (validation / checkpointing)
        try:
            tb.post_epoch_process(epoch)
        except Exception:
            pass

if __name__ == "__main__":
    main()