# tb-adaption-dev

### Parallel Transfers

The `send_files_to_aria()` function now accepts a `num_workers` argument to
allow multiple DICOM associations to send files in parallel.  Increasing the
number of workers can significantly reduce upload time when the network and
server permit concurrent connections:

```python
from export import send_files_to_aria
send_files_to_aria(files, num_workers=4)
```
 
