file_name = './test_data'

f = open(file_name, 'r', encoding='utf-8')
lines = f.readlines()
f.close()

f = open(file_name, 'w', encoding='ANSI')
for line in lines:
    f.write(line)
f.close()