import numpy

l = 89200
rule = 20

lines = open('train.txt','r').readlines()[:89200]
lines = numpy.split(numpy.array(lines), 20)

for instances in lines:
    ents = set()
    for l in instances:
        e = l.split()[0]
        ents.add(e)
        if e in ents:
            print(ents)
            print(e)
    if len(ents) != 240:
        print(len(ents))
        print(ents)