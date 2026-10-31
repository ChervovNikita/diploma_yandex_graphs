import os,resource,sys
resource.setrlimit(resource.RLIMIT_AS,(24*1024**3,24*1024**3))
os.execv(sys.argv[1],sys.argv[1:])
