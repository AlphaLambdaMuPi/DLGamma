from settings import *
import pickle
from os.path import join as pjoin
from collections import Counter

def generate_all(name, max_occ=5, min_len=10, max_len=30):
    with open(DATA['word_list_test'], 'rb') as f:
        st = set(pickle.load(f))

    c = Counter()
    path = pjoin(PATH['train_data'], name)
    if not os.path.isdir(path):
        os.mkdir(path)

    train_file = pjoin(path, 'train.txt')
    desc_file = pjoin(path, 'desc.txt')
    test_file = pjoin(path, 'test.txt')
    with open(DATA['train_sentences']) as f, \
         open(train_file, 'w') as fw:
        mx = 0
        for ln in f:
            ls = ln.strip('\n').split()
            for i, x in enumerate(ls):
                if i > max_len: break
                if i < min_len: continue
                if x in st and c[x] < max_occ:
                    s = ' '.join(ls[:i])
                    s += ' [{}]\n'.format(x)
                    mx = max(i, mx)
                    fw.write(s)
                    c[x] += 1

    with open(DATA['test_sentences']) as f, \
         open(DATA['test_answers']) as fa, \
         open(test_file, 'w') as fw:
        while True:
            ln = f.readline()
            if not ln: break
            ls = ln.strip('\n').split()
            ls[0] = '<s>'
            opt = []
            for i, x in enumerate(ls):
                if x[0] == '[' and x[-1] == ']':
                    idx = i
                    opt.append(x[1:-1])
                    break
            for i in range(4):
                lns = f.readline().strip('\n').split()
                opt.append(lns[idx][1:-1])

            mx = max(mx, idx)
            sa = fa.readline().strip('\n').split()
            ans = sa[idx][1:-1]
            ans_idx = opt.index(ans)
            opt[ans_idx], opt[0] = opt[0], opt[ans_idx]
            fw.write('{} [{}]\n'.format(' '.join(ls[:idx]), '@'.join(opt)))
            

    with open(desc_file, 'w') as fw:
        fw.write('Maxlen = {}'.format(mx))


def generate_bidirect_all(name, max_occ=5, min_len=10, max_len=30):
    with open(DATA['word_list_test'], 'rb') as f:
        st = set(pickle.load(f))

    c = Counter()
    path = pjoin(PATH['train_data'], name)
    if not os.path.isdir(path):
        os.mkdir(path)

    train_file = pjoin(path, 'train.txt')
    desc_file = pjoin(path, 'desc.txt')
    test_file = pjoin(path, 'test.txt')
    with open(DATA['train_sentences']) as f, \
         open(train_file, 'w') as fw:
        mx = 0
        for ln in f:
            ls = ln.strip('\n').split()
            for i, x in enumerate(ls):
                if i > max_len or len(ls)-i-1 < min_len: break
                if i < min_len or len(ls)-i-1 > max_len: continue
                if x in st and c[x] < max_occ:
                    s = ' '.join(ls[:i])
                    s += ' [{}]\n'.format(x)
                    mx = max(i, mx)
                    mx = max(len(ls)-i-1, mx)
                    fw.write(s)
                    s = ' '.join(ls[i+1:][::-1])
                    fw.write(s + '\n')
                    c[x] += 1

    with open(DATA['test_sentences']) as f, \
         open(DATA['test_answers']) as fa, \
         open(test_file, 'w') as fw:
        while True:
            ln = f.readline()
            if not ln: break
            ls = ln.strip('\n').split()
            ls[0] = '<s>'
            ls.append('</s>')
            opt = []
            for i, x in enumerate(ls):
                if x[0] == '[' and x[-1] == ']':
                    idx = i
                    opt.append(x[1:-1])
                    break
            for i in range(4):
                lns = f.readline().strip('\n').split()
                opt.append(lns[idx][1:-1])

            mx = max(mx, idx)
            mx = max(mx, len(ls) - idx - 1)
            sa = fa.readline().strip('\n').split()
            ans = sa[idx][1:-1]
            ans_idx = opt.index(ans)
            opt[ans_idx], opt[0] = opt[0], opt[ans_idx]
            fw.write('{} [{}]\n'.format(' '.join(ls[:idx]), '@'.join(opt)))
            fw.write(' '.join(ls[idx+1:]) + '\n')
            

    with open(desc_file, 'w') as fw:
        fw.write('Maxlen = {}'.format(mx))
