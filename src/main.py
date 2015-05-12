#!/usr/bin/env python3
import sys, os
from settings import *
import argparse
import importlib

class Main():
    def __init__(self):
        parser = argparse.ArgumentParser()
        subpars = parser.add_argument('profile', type=str, nargs='?')


        args = parser.parse_args()

        if args.profile is not None:
            self.run(args.profile)
        else:
            s = input('Input profile name: ')
            self.run(s)

    def run(self, profile):
        P = importlib.import_module('profile_scripts.{}'.format(profile))
        ex = P.Profile()
        try:
          ex.start()
        except KeyboardInterrupt:
          pass
        finally:
          ex.end()

def main():
    Main()

main()
