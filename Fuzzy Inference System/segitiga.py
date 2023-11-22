def segitiga(x, a, b, c):
    if a > b or a > c:
        raise ValueError('Ingat bahwa a <= b, a <= c')
    elif b > c:
        raise ValueError('Ingat bahwa a <= b, b <= c')

    y = [0] * len(x)

    for i in range(len(y)):
        if x[i] <= a or x[i] >= c:
            y[i] = 0
        if a != b and a < x[i] < b:
            y[i] = (x[i] - a) / (b - a)
        if x[i] == b:
            y[i] = 1
        if b != c and b < x[i] < c:
            y[i] = (c - x[i]) / (c - b)

    return y

# Contoh penggunaan
x_values = [0.5, 1.5, 2.5]
a_value = 1
b_value = 2
c_value = 3

result = segitiga(x_values, a_value, b_value, c_value)
print(result)
