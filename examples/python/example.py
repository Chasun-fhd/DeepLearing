def echo(*input, **kwargs):
    print(input, type(input))
    print(kwargs, type(kwargs))


echo("aaa",{"name":"lily"}, name="aab")