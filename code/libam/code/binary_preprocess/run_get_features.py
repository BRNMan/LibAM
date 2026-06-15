import subprocess
import os
import time

IDA_DATABASE_FOLDER = '/nethome/mbraun39/michaels_firmware'

start_total = time.perf_counter()
for path in os.listdir(IDA_DATABASE_FOLDER):
    if path.endswith(".i64"):
        start_iter = time.perf_counter()
        shortpath = path.split("/")[-1].split(".")[:-1]
        if len(shortpath) > 0:
            shortpath = ".".join(shortpath)
        else:
            shortpath = shortpath[0]
        print(f"Turning {path} into {shortpath}.json")
        retval = subprocess.run(f"ida -A -S'/nethome/mbraun39/ida_get_func_call.py /nethome/mbraun39/fcg/{shortpath}.json' '{IDA_DATABASE_FOLDER}/{path}'", shell=True)
        end_iter = time.perf_counter()
        print(f"Completed with status {retval.returncode} after {end_iter-start_iter:.2f} seconds.")\

end_total = time.perf_counter()
print(f"Total time: {end_total - start_total:.2f} seconds.")
