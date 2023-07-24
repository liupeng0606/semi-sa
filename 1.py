s = [1, 2]



x = iter(s)
for i in range(10):
    try:
        v = next(x)
    except:
        x = iter(s)
        # v = next(x)

    print(v)

    


    