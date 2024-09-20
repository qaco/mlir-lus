The sub-folders correspons to:
* benchmarks - ML benchmarks used in papers -- their original form, without mlirlus-specific modifications.
* hept - Lustre/Heptagon manual reencoding of the benchmaks, used for ergonomy and performance comparisons.
* iree - mlirlus output for the IREE back-end (CPU and GPU)
* llvmir - mlirlus output not using the iree extensions. Execution relies on a minimalist CPU-based runtime we wrote, and Lus nodes are implemented each as concurrent infinitely running processes.