#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Megatron Cython Build"""

import os
import sys

from Cython.Build import cythonize
from Cython.Compiler import Options
Options.error_on_unknown_names = False

from distutils.core import setup
from Cython.Build import cythonize

group_num = 2
file_groups = []


v3_list = ["megatron/core/optimizer/d2d_distrib_optimizer.py",
           "megatron/core/optimizer/d2d_optimizer_comm.py"]
v2_list = []


def build_file_groups():
    """build file groups for cython build"""
    group_dict_v3 = dict()
    group_dict_v2 = dict()

    group_dict_v3.update({"file_list": v3_list})
    group_dict_v3.update({"version": "3"})
    file_groups.append(group_dict_v3)


    group_dict_v2.update({"file_list": v2_list})
    group_dict_v2.update({"version": "2"})
    file_groups.append(group_dict_v2)


def exec_cython_build():
    """exec cython build"""
    for file_group in file_groups:
        version = file_group["version"]
        for file in file_group["file_list"]:
            if not os.path.exists(file):
               continue
            if not file.endswith(".py"):
               print ("%s not python source file"%file)
               continue

            pure_file_name = file[:-3]
            dir_path = os.path.dirname(file)
            
            setup(name=pure_file_name,
                  packages=["megatron", "megatron.core", "megatron.core.optimizer"],
                  ext_modules=cythonize(file, compiler_directives={'language_level' : version}))

            os.system("rm -rf %s.py"%pure_file_name)
            os.system("rm -rf %s.c"%pure_file_name)

            build_path = remove_first_directory(pure_file_name)
            build_path = "build/lib.linux-x86_64-*/%s*"%remove_first_directory(pure_file_name)

            os.system("cp %s %s"%(build_path, dir_path))

def clear_build_dir():
    """rm build dir"""
    if os.path.exists("./build"):
       os.system("rm -rf ./build")

def remove_first_directory(path):
    parts = path.split('/')  
    return '/'.join(parts[1:])

def show_help():
    """show help infomations."""
    print("python cython_setup.py -h or --help: show help infomations.")
    print("python cython_setup.py build: exec cython build.")

def main():
    """main"""
    if len(sys.argv) < 2:
        print (f"Number of command line parameters is not right,"
               f"should add 'build', like 'python cython_setup.py build'")
        return

    if len(sys.argv) > 2:
        print ("Number of command line parameters is not right, more than the normal numbers")
        return

    if sys.argv[1] != "build" and sys.argv[1] != "-h" and sys.argv[1] != "--help":
        print ("The last parameter is not right, should be build,or -h,or --help.")
        return

    if sys.argv[1] == "-h" or sys.argv[1] == "--help":
        show_help()
        return

    build_file_groups()
    exec_cython_build()
    clear_build_dir()

if __name__ == '__main__':
    main()

