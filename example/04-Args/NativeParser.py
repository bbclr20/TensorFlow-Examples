import argparse

parser = argparse.ArgumentParser()
parser.prog = "NativeParser"
parser.usage = "NativeParser.py [-h] [-f] [-v {1, 2, 3}] -d dir"
description = """
              This is a simple demo of arg parser which
              can be used to parse the input arguments.
              """
parser.description = description

parser.add_argument("-f", "--filename", type=str, default="default.txt",
                    help="Input file name")
parser.add_argument("-v", "--version", choices=[1, 2, 3], type=int, default=1,
                    help="API version")
parser.add_argument("-d", "--dir", type=str, required=True,
                    help="Output directory")

args = parser.parse_args()
print("Filename: ", args.filename)
print("Version: ", args.version)
print("Output directory: ", args.dir)

# command line example:
# $ python3 -d output -v 2
# $ python3 -d output