import subprocess
import os

for path in os.listdir('./idb_files'):
    if path.endswith(".i64"):
        shortpath = path.split("/")[-1].split(".")[:-1]
        if len(shortpath) > 0:
            shortpath = ".".join(shortpath)
        else:
            shortpath = shortpath[0]
        print(f"Running {path} for {shortpath}.json")
        retval = subprocess.run(f"ida -A -S'/nethome/mbraun39/ida_get_features.py /nethome/mbraun39/features/{shortpath}.json' '/nethome/mbraun39/idb_files/{path}'", shell=True)
        print(f"Completed with status {retval.returncode}")

