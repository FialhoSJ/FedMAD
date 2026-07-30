import py_compile, sys
py_compile.compile(sys.argv[1], doraise=True)
print(sys.argv[1].split("/")[-1], "OK")
