# nmc_scheduling
Scheduling algorithm for Neuromatch conference

I'm releasing this repository as-is without turning it into a nice package because I figure
it's better that it's available than that it's never released because I don't have the time
to make it perfect. Please let me know if you need any help using it. It won't actually run
in its current state because you need access to data files that I can't make publicly
available, but at least you can see the ideas.

The key file you want to look at is ``scheduler_sessions.py``.

Installation:

```
conda create -n nmc_scheduling python=3 numpy scipy matplotlib pandas seaborn
conda activate nmc_scheduling
pip install mip
```
