# ancv - OpenCV RPC Server

```
$ ./ancv --help

NAME:
   ancv - by gocv

USAGE:
   ancv [global options] command [command options] [arguments...]

COMMANDS:
   rpc, rpc-serve  Start a rpc serve
   det, detect     Detect objects
   rec, recognize  Recognize Faces
   help, h         Shows a list of commands or help for one command

GLOBAL OPTIONS:
   --help, -h  show help (default: false)
```

```
$ ./ancv rpc --help

NAME:
   ancv rpc - Start a rpc serve

USAGE:
   ancv rpc [command options] [arguments...]

OPTIONS:
   --host value  ip:port
   --help, -h    show help (default: false)
```

```
$ ./ancv det --help

NAME:
   ancv det - Detect objects

USAGE:
   ancv det [command options] [arguments...]

OPTIONS:
   --input value, -i value   input image
   --output value, -o value  output image
   --coco-dir value          SsdLite coco dir
   --help, -h                show help (default: false)
```

```
$ ./ancv rec --help

NAME:
   ancv rec - Recognize Faces

USAGE:
   ancv rec [command options] [arguments...]

OPTIONS:
   --input value, -i value    input image
   --output value, -o value   output image
   --storage value, -s value  storage dir
   --recognized value         recognized file
   --lbpcascades value        lbpcascades dir
   --faces-dir value          faces image dir
   --ask, -a                  Ask label for unrecognized (default: false)
   --help, -h                 show help (default: false)
```
