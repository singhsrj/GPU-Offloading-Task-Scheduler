# GPU Offloading Task Scheduler
<table>
<tr>
<td>
  <img width="768" height="512" 
       alt="40b20df9-49a1-43d2-8cdd-5672ef0c0398" 
       src="https://github.com/user-attachments/assets/3ed94926-52a8-4517-b19e-a136a0d24822" 
       style="border-radius:10px; box-shadow: 0 4px 8px rgba(0,0,0,0.15);" />
</td>
<td style="vertical-align: middle; padding-left: 10px;">
  An intelligent task scheduler that uses a machine learning model to dynamically decide when to offload computationally intensive tasks from the CPU to the GPU.  
  <br><br>
  Its goal is to improve overall system throughput by using hardware resources more effectively.
</td>
</tr>
</table>


---

## System Architecture

The system uses a hybrid user-space and kernel-space model to separate high-level decision-making from low-latency system control. This design ensures that complex logic doesn't run inside the kernel, which enhances system stability and flexibility.


###  User-Space Daemon

It's a standard user application responsible for all complex, non-time-critical operations.

* **Machine Learning Inference:** Loads and executes the pre-trained ML model to decide if a given task is a good candidate for GPU offloading. This decision is context-aware, considering both the **nature of the task** (e.g., its computational pattern) and the real-time **state of the system** (e.g., current CPU/GPU load, memory pressure).
* **Policy Management:** Manages the high-level rules for scheduling, such as which applications are eligible for offloading.
* **GPU Queue Management:** Maintains the queue of tasks approved for offloading and submits them to the GPU via the CUDA runtime.
* **Communication:** Interacts with the kernel module to receive task information and send back scheduling decisions.

### Kernel-Space Module

This component is a Loadable Kernel Module (LKM) that acts as the **"hands"** of the scheduler. It's designed to be minimal and fast, serving as a low-level observation and enforcement agent.

* **Intercepting Scheduler Events:** Uses **Kernel Tracepoints** to non-intrusively monitor the kernel's task scheduling decisions in real-time. Specifically, it attaches to the `sched_switch` tracepoint to see which task the CPU plans to run next.
* **Communication:** Communicates with the user-space daemon via a **Netlink socket**. When a new, potentially offloadable task is about to be scheduled, the module sends key information (e.g., process ID, application name) to the daemon.
* **Enforcing Decisions:** Receives commands from the daemon. If a task is to be offloaded, the kernel module will alter that task's state to `TASK_INTERRUPTIBLE`, effectively pausing it and removing it from the CPU's run queue while it awaits GPU processing.

---

## Technology Stack & Workloads

The project will utilize both OpenMP and CUDA to create and manage the computational workloads

* **CUDA:** The portions of code that are to be executed on the GPU will be written in CUDA. This provides direct, high-performance access to the GPU's parallel processing capabilities.
* **OpenMP:** We will use OpenMP directives in our C/C++ test applications to identify and delineate CPU-bound, parallelizable sections of code (`#pragma omp parallel for`). These marked sections are the primary signals for our scheduler to identify a "task" that could potentially be offloaded.

---

## Performance Benchmarking

To validate the scheduler's effectiveness, we'll measure performance by comparing the execution time of test workloads under two scenarios:

1.  **Baseline:** The application runs normally on the CPU, utilizing OpenMP for parallelism.
2.  **With Offloading:** Our scheduler is active, dynamically offloading designated tasks to the GPU.

The key metrics for evaluation will be:

* **Total Execution Time:** The primary measure of performance improvement.
* **CPU and GPU Utilization:** To ensure that hardware resources are being used effectively and the CPU is successfully freed up for other tasks.
* **Scheduler Overhead:** We'll also quantify the latency introduced by the decision-making loop (kernel -> user-space -> kernel) to understand its cost.

Test workloads will consist of common high-performance computing tasks such as matrix multiplication, image processing filters, and physics simulations.
