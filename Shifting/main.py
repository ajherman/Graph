v=5
k=3

x = [0]*v

def generate_binary_strings(n, s=''):
    if n == 0:
        print(s)
    else:
        generate_binary_strings(n-1, s + '0')
        generate_binary_strings(n-1, s + '1')

v = 3  # Change this to the desired length
generate_binary_strings(v)