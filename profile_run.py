"""
profile_run.py — drop this next to test1.py and run:
    python profile_run.py

After you close the viewer window, it prints the top 40 hottest call sites
sorted by cumulative time, then saves a full dump to profile.out which you
can inspect later with:
    python -m pstats profile.out
"""

import cProfile
import pstats
import io

pr = cProfile.Profile()
pr.enable()

import test_modular
test_modular.main()

pr.disable()

# Save raw dump for later
pr.dump_stats('profile.out')

# Print top 40 by cumulative time
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(40)
print(s.getvalue())

# Also print top 40 by total self-time (best for finding hot inner loops)
s2 = io.StringIO()
ps2 = pstats.Stats(pr, stream=s2).sort_stats('tottime')
ps2.print_stats(40)
print("\n\n===== SORTED BY SELF TIME (inner loops) =====")
print(s2.getvalue())
