def trapesium(x, a, b, c, d):
    if a > b or a > c or a > d:
        raise ValueError('Ingat bahwa a <= b, a <= c, a <= d')
    elif b > c or b > d:
        raise ValueError('Ingat bahwa a <= b, b <= c, b <= d')

    y = [0 if xi <= a or xi >= d else
         0 if a != b and a < xi < b else
         1 if b <= xi <= c else
         (d - xi) / (d - c) if c != d and c < xi < d else 0
         for xi in x]

    return y

# Contoh penggunaan
x_values = [0.5, 1.5, 2.5, 3.5]
a_value = 1
b_value = 2
c_value = 3
d_value = 4

result = trapesium(x_values, a_value, b_value, c_value, d_value)
print(result)
