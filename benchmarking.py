from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments

args = PyTorchBenchmarkArguments(models=["allenai/longformer-base-4096"], batch_sizes=[1,2], sequence_lengths=[4096], inference=False, training=True, train_memory_csv_file="plots_pt/training_mem_fp16.csv", save_to_csv=True, env_print=False, fp16=True)
benchmark = PyTorchBenchmark(args)
results = benchmark.run()
print(results)