
nb_not_match = 0
not_match = []
not_match2 = []

with open("a.raw.txt") as f1:
    with open("data/java-small-preprocessed-code2vec/java-small/java-small.val.c2v") as f2:
        ls1 = f1.readlines()
        ls2 = f2.readlines()
        for i, l in enumerate(ls2):
            if l not in ls2:
                not_match.append(l)
                

print(len(not_match))

