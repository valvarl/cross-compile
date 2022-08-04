# cross-compile
Script for cross-compilation and collection inference statistics. It creates the following directories

* `models/` created with loading models
* `logs/` contains AutoTVM tuning history

Tuning option:
```
python cross-compile.py tune atvm -m mace_mobilenet_v1 -k "huawei"
```
Execution option:
```
python cross-compile.py exec -m "huawei.mace_mobilenet_v1.float32.atvm.so" -k "htc"
```
Statistics on the output time of precompiled networks on various devices is available [here](https://docs.google.com/spreadsheets/d/1ooEsjJjK94f29Yyz2lto9aOlKAv27fwdClhDWXIVpKI/edit?usp=sharing).
