def f(x, y, **kwargs):
    print(f"{x}->{kwargs['var1']}")

# f(1,2, **{'var1':10, 'var2':20})

def g(**kwargs):
    a = kwargs
    f(100, 2, **a)

g(**{'var1':10})

