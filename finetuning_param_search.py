import subprocess

script_path = "mm_okvqa_finetuning.py"
args = [
    "--model", "OFA-Sys/ofa-base",
    "--target_model", "ofa",
    "--location_encoding", "none",
    "--lr", "2e-5",
    "--opt_wd", "1e-5",
    "--batch_size", "4",
    "--max_steps", "2000",
    "--accumulate_grad_batches", "2",
    "--run_name", "ofa_base_okvqa",
    "--train",
    "--evaluate",
    "--source", "vinvl",
    "--iteration", "0",
    "--root", "/gaueko0/users/ietxarri010/GrAL_Irene/okvqa"
]

iterkop = 10

for i in range(iterkop):
    try:
        iter_index = args.index("--iteration") + 1
        args[iter_index] = str(i)

    except ValueError:
        args += ["--iteration", str(i)]

    subprocess.run(["python", script_path] + args, check=True)