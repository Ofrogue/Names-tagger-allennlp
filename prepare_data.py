import os

directory = "./data/names"


data_list = list()

for filename in os.listdir(directory):
    text = open(os.path.join(directory, filename)).read()
    names = text.split()
    lang = filename.split('.')[0]
    data_list.extend([name+' '+lang for name in names])


f = open('data/output.txt', 'w')
f.write('\n'.join(data_list))
f.close()


