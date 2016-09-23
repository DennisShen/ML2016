import sys

# initialize
data    = []
count   = 0
col_num = 0

# col num
temp_file = open(sys.argv[2], 'r')
temp_line = temp_file.readline().strip().split(' ')
col_num   = len(temp_line)
temp_file.close()

# import data
for line in file(sys.argv[2], 'r'):
    for num in line.strip().split(' '):
        if count % col_num == int(sys.argv[1]):
            data.append(float(num))
        count = count + 1        

# sort data
data_s = sorted(data)

# output answer
num = 0
o_file = open('ans1.txt', 'w')
for s in data_s:
    num = num + 1
    if num != len(data_s):
        o_file.write(str(s) + ',')
    else:
        o_file.write(str(s))
