from PIL import ABLATION3_RUNS
import os
import subprocess
import sys

SCRIPT = os.path.abspath("/home/chandan22140/test_current/train_vit_rotational.py")

# General params
MODEL = "google/vit-base-patch16-224-in21k"
SEED = "42"

def construct_run(label, log_file, dataset, method, rank, extra_args):
    """Factory to construct args based on optimal hyperparams mapped from table."""
    lr = "0.0002"
    wd = "0.0"
    epochs = "10"
    bs = "32"
    
    # Handle the fact that extra_args might be modified
    args_copy = extra_args[:]
    
    # Defaults mapping from table
    if method == "v1":
        if dataset == "cifar100":
            lr = "0.00015"
            wd = "0.00027"
            epochs = "10"
            bs = "32"
            args_copy.append("--orthogonality-weight=0.046")
        elif dataset == "fgvc_aircraft":
            lr = "0.00091"
            wd = "0.000019"
            epochs = "30"
            bs = "32"
            args_copy.append("--orthogonality-weight=0.0019")
        args_copy.append("--regularization-type=frobenius")
        
    elif method == "V2" and "--use-butterfly" not in extra_args:
        # V2a (Givens)
        if dataset == "cifar100":
            lr = "0.00043"
            wd = "0.0"
            epochs = "10"
            bs = "32"
        elif dataset == "fgvc_aircraft":
            lr = "0.0033"
            wd = "0.0"
            epochs = "30"
            bs = "32"
            
    elif method == "V2" and "--use-butterfly" in extra_args:
        # V2b (Butterfly)
        if dataset == "cifar100":
            lr = "0.0002"
            wd = "0.0"
            epochs = "10"
            bs = "16"
        elif dataset == "fgvc_aircraft":
            lr = "0.0019"
            wd = "0.0"
            epochs = "30"
            bs = "32"

    args = [
        f"--dataset={dataset}",
        f"--method={method}",
        f"--rank={rank}",
        f"--model={MODEL}",
        f"--seed={SEED}",
        f"--learning-rate={lr}",
        f"--weight-decay={wd}",
        f"--epochs={epochs}",
        f"--batch-size={bs}",
        "--lr-ratio-s=10.0",
        "--no-wandb",
    ] + args_copy
    return (label, log_file, args)


ABLATION1_RUNS = [
    construct_run("A1_UV_cifar100", "ablation1_UV_cifar100.log", "cifar100", "v1", 16, ["--rotation-side=both"]),
    construct_run("A1_UV_fgvc", "ablation1_UV_fgvc.log", "fgvc_aircraft", "v1", 16, ["--rotation-side=both"]),
    construct_run("A1_U_only_cifar100", "ablation1_U_only_cifar100.log", "cifar100", "v1", 16, ["--rotation-side=u_only"]),
    construct_run("A1_U_only_fgvc", "ablation1_U_only_fgvc.log", "fgvc_aircraft", "v1", 16, ["--rotation-side=u_only"]),
    construct_run("A1_V_only_cifar100", "ablation1_V_only_cifar100.log", "cifar100", "v1", 16, ["--rotation-side=v_only"]),
    construct_run("A1_V_only_fgvc", "ablation1_V_only_fgvc.log", "fgvc_aircraft", "v1", 16, ["--rotation-side=v_only"]),
    construct_run("A1_UV_matched_cifar100", "ablation1_UV_matched_cifar100.log", "cifar100", "v1", 11, ["--rotation-side=both"]),
    construct_run("A1_UV_matched_fgvc", "ablation1_UV_matched_fgvc.log", "fgvc_aircraft", "v1", 11, ["--rotation-side=both"]),
]

ABLATION2_RUNS = [
    construct_run("A2_V1_full_cifar100", "ablation2_V1_full_cifar100.log", "cifar100", "v1", 16, []),
    construct_run("A2_V1_full_fgvc", "ablation2_V1_full_fgvc.log", "fgvc_aircraft", "v1", 16, []),
    construct_run("A2_V1_frozen_cifar100", "ablation2_V1_frozen_cifar100.log", "cifar100", "v1", 16, ["--freeze-s"]),
    construct_run("A2_V1_frozen_fgvc", "ablation2_V1_frozen_fgvc.log", "fgvc_aircraft", "v1", 16, ["--freeze-s"]),
    construct_run("A2_V2b_full_cifar100", "ablation2_V2b_full_cifar100.log", "cifar100", "V2", 16, ["--use-butterfly"]),
    construct_run("A2_V2b_full_fgvc", "ablation2_V2b_full_fgvc.log", "fgvc_aircraft", "V2", 16, ["--use-butterfly"]),
    construct_run("A2_V2b_frozen_cifar100", "ablation2_V2b_frozen_cifar100.log", "cifar100", "V2", 16, ["--use-butterfly", "--freeze-s"]),
    construct_run("A2_V2b_frozen_fgvc", "ablation2_V2b_frozen_fgvc.log", "fgvc_aircraft", "V2", 16, ["--use-butterfly", "--freeze-s"]),
]


def launch(command, log_path):
    log_file = open(log_path, "w")
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, log_file

def main():
    # ALL_RUNS = ABLATION1_RUNS 
    ALL_RUNS = ABLATION1_RUNS + ABLATION2_RUNS + ABLATION3_RUNS
    
    processes = []
    
    for label, log_path, extra_args in ALL_RUNS:
        command = ["python", SCRIPT, *extra_args]
        # print(command)
        # print()
        # print()
        
        process, log_file = launch(command, log_path)
        processes.append((label, process, log_file, log_path))
        print(f"Started {label} (PID={process.pid})  →  {log_path}")
    
    print(f"\nLaunched {len(processes)} processes total.")
    print("Logs: tail -f ablation*.log")

if __name__ == "__main__":
    main()
