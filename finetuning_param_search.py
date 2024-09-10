import subprocess

script_path = "mm_okvqa_finetuning.py"
args = [
    "--model", "OFA-Sys/ofa-base",
    "--target_model", "ofa",
    "--location_encoding", "none",
    "--lr", "2e-5",
    "--opt_wd", "0",
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

iterkop = 5

#wd = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
for i in range(3, iterkop):
    try:
        iter_index = args.index("--iteration") + 1
        args[iter_index] = str(i)

    except ValueError:
        args += ["--iteration", str(i)]

    """"
    try:
        iter_index = args.index("--opt_wd") + 1
        args[iter_index] = str(wd[i])

    except ValueError:
        args += ["--opt_wd", str(i)]
    """

    subprocess.run(["python", script_path] + args, check=True)