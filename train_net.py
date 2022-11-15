# Imports
import torch
import torch.multiprocessing as mp
import argparse, random, os, signal, subprocess
import submitit
from pathlib import Path

def parse_args():
	parser = argparse.ArgumentParser(description='Template')
	parser.add_argument('-gpus', type=str, default="0", help='GPUs list')
	parser.add_argument('-slurm_ngpus', type=int, default = 8, help='num of gpus per node')
	parser.add_argument('-slurm_nnodes', type=int, default = 2, help='number of nodes')
	parser.add_argument('-slurm_partition', type=str, default = "general", help='slurm partition')
	parser.add_argument('-slurm_timeout', type=int, default = 2800, help='slurm timeout minimum')
	args = parser.parse_args()
	return args

args = parse_args()
args.port = random.randint(49152,65535)

class SLURM_Trainer(object):
	def __init__(self, args):
		self.args = args

	def __call__(self):
		args.ngpus_per_node = torch.cuda.device_count()

		# requeue job on SLURM preemption
		signal.signal(signal.SIGUSR1, handle_sigusr1)
		signal.signal(signal.SIGTERM, handle_sigterm)

		# find a common host name on all nodes
		cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
		stdout = subprocess.check_output(cmd.split())
		host_name = stdout.decode().splitlines()[0]
		self.args.dist_url = f'tcp://{host_name}:{self.args.port}'

		# distributed parameters
		self.args.rank = int(os.getenv('SLURM_NODEID')) * self.args.ngpus_per_node
		self.args.world_size = int(os.getenv('SLURM_NNODES')) * self.args.ngpus_per_node
		
		job_env = submitit.JobEnvironment()
		args.output_dir = Path(str(args.output_dir).replace("%j", str(job_env.job_id)))
		args.gpu = job_env.local_rank
		args.rank = job_env.global_rank
		
		train(None, self.args)

# Signal Handlers
def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()

def handle_sigterm(signum, frame):
    pass
		
args.output_dir = "path/to/shared/output/directory"
executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=30)

executor.update_parameters(
	mem_gb=12*args.slurm_ngpus,
	gpus_per_node=args.slurm_ngpus,
	tasks_per_node=args.slurm_ngpus,
	cpus_per_task=2,
	nodes=args.slurm_nnodes,
	timeout_min=args.slurm_timeout,
	slurm_partition=args.slurm_partition
)

executor.update_parameters(name="Template")
trainer = SLURM_Trainer(args)
job = executor.submit(trainer)
print(f"Submitted job_id: {job.job_id}")