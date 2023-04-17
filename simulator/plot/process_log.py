import json
import csv
import copy

with open('trace_alloc.csv', 'w') as f2:
    f_csv = csv.writer(f2, delimiter='\t')

    data = []
    for i in ['easyscale_homo', 'easyscale']:
    #for i in ['fifo', 'easyscale_homo', 'easyscale']:
        with open('../summary_{}.json'.format(i), 'r') as f:
            summary =json.load(f)

            header = []
            v100 = []
            p100 = []
            t4 = []
            total = []

            for item in summary:
                header.append(item['timestamp'])
                v100.append(item['gpu_nums']['v100'])
                p100.append(item['gpu_nums']['p100'])
                t4.append(item['gpu_nums']['t4'])
                total.append(item['gpu_nums']['v100']+item['gpu_nums']['p100']+item['gpu_nums']['t4'])

            data.append(copy.deepcopy(total))
    
    f_csv.writerow(["timestamp", "EasyScale_{homo}", "EasyScale_{heter}"])
    #f_csv.writerow(["timestamp", "YARN-CS", "EasyScale_{homo}", "EasyScale_{heter}"])
    max_len = max([len(d) for d in data])
    for i in range(len(data)):
        if len(data[i]) < max_len:
            data[i] = data[i] + [0] * (max_len-len(data[i]))
    for i in range(max_len):
        f_csv.writerow([i*60] + [d[i] for d in data])
    

def find_value(line, start, end):
    value = None
    if line.find(start) != -1:
        first = line.find(start) + len(start)
        tmp_line = line[first:]
        if len(end) == 0:
            value = tmp_line
        elif tmp_line.find(end) != -1:
            end = tmp_line.find(end)
            value = tmp_line[:end]
        if len(value) > 10:
            value = value[:9]
    return value

with open('jct-makespan.csv', 'w') as f2:
    f_csv = csv.writer(f2, delimiter='\t')
    f_csv.writerow(["#server",	"JCT", "Makespan"])
    for i in ['fifo', 'easyscale_homo', 'easyscale']:
        with open('../trace_log/log_{}'.format(i), 'r') as f:
            lines = f.readlines()[-8:]
            makespan = float(find_value(line=lines[0], start="SIMULATOR TIME: ", end=" -")) - 60
            jct = float(find_value(line=lines[-3], start="Average JCT: ", end=""))
        if i == "fifo":
            f_csv.writerow(['YARN-CS', jct, makespan])
        elif i == "easyscale":
            f_csv.writerow(['EasyScale_{heter}', jct, makespan])
        elif i == "easyscale_homo":
            f_csv.writerow(['EasyScale_{homo}', jct, makespan])


