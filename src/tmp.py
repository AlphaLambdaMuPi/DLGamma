f = open('cos.txt')
lst = [float(x) for x in f]

g = open('outcos.txt', 'w')
g.write('id,answer\n')
eng = 'abcde'
for i in range(len(lst)//5):
    best_score = -111
    ansid = -1
    for j in range(5):
        score = lst[i*5+j]
        if score > best_score:
            best_score = score
            ansid = j
    g.write('{},{}\n'.format(i+1, eng[ansid]))
g.close()

