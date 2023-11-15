    
import itertools


def is_even_loop(word, left, right):
    max_color = -1
    for j in range(left, right + 1, 1):
        max_color = max(max_color, word[j])
    
    if max_color % 2 == 1:
        return False
    else:
        return True
    
def get_membership(num_colors, word):
    positions = []
    for i in range(num_colors):
        positions.append((-1, -1))
    
    loop_mask = 0
    for i, color in enumerate(word):
        lft, rgt = positions[color]
        has_loop = False
        if lft == -1:
            positions[color] = (i, -1)
        elif rgt == -1:
            positions[color] = (lft, i)
            has_loop = True
        else:
            positions[color] = (rgt, i)
            has_loop = True
        
        lft, rgt = positions[color]
        is_even = is_even_loop(word, lft, rgt)
        #print("lft = " + str(lft) + ", rgt=" + str(rgt) + ", is_even=" + str(is_even) + ", has_loop=" + str(has_loop))
        if has_loop and is_even:
            loop_mask |= 2
        elif has_loop and not is_even:
            loop_mask |= 1
        #print("loop_mask=" + str(loop_mask))
        if loop_mask >= 3:
            return 0
    
    if loop_mask == 2:
        return 1
    elif loop_mask == 1:
        return -1
    else:
        return 0


def get_all_data(num_colors, length):
    # we verify by enumerating all possible words
    meta_elems = []
    for i in range(length):
        meta_elems += [range(num_colors)]
        
    # enumerate all possible words
    perms = [list(p) for p in itertools.product(*meta_elems)]
    pos = list(filter(lambda x: get_membership(num_colors, x) == 1, perms))
    negs = list(filter(lambda x: get_membership(num_colors, x) == -1, perms))
    doncares = list(filter(lambda x: get_membership(num_colors, x) == 0, perms))
    
    return (pos, negs, doncares)            
          


pos = []
negs = []
dontcares = []
num_colors = 6
length = 9


pos, negs, dontcares = get_all_data(num_colors, length)
# print("Positive: ")
# for p in pos:
#     print(p)

# print("Negative: ")
# for p in negs:
#     print(p)
    
# print("Dontcares: ")
# for p in dontcares:
#     print(p)

with open('data' + str(num_colors) +'-' + str(length) + '-all.txt', 'w') as the_file:
    the_file.write(str(len(pos) + len(negs)) + ' ' + str(num_colors) + '\n')
    for p in pos:
        the_file.write('1 ' + str(len(p)))
        for i in p:
             the_file.write(' ' + str(i))
        the_file.write('\n')

    for p in negs:
        the_file.write('0 ' + str(len(p)))
        for i in p:
             the_file.write(' ' + str(i))
        the_file.write('\n')
        
    for p in dontcares:
        the_file.write('-1 ' + str(len(p)))
        for i in p:
             the_file.write(' ' + str(i))
        the_file.write('\n')

print("#colors=" + str(num_colors))
print("#length=" + str(length))    
print("#pos=" + str(len(pos)))
print("#neg=" + str(len(negs)))
print("#dont=" + str(len(dontcares)))