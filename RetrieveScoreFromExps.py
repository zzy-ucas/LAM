scores = [0] * 8
inds = [6, 3, 2, 1, 0, 5, 4, 7]

with open('ucm.txt', 'r') as f:
    for ind, l in enumerate(f.readlines()):
        l = l.replace('\n', '').split(': ')[-1].replace('\r', '')
        scores[inds[ind]] = float(l)
print(scores)
