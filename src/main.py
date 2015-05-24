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
        if profile == 'list':
            ls = os.listdir(os.path.join(PATH['src'], 'profile_scripts'))
            for f in filter(lambda x: x[-3:] == '.py', ls):
                P = importlib.import_module('profile_scripts.{}'.format(f[:-3]))
                try:
                    ex = P.Profile()
                    print('{} : {}'.format(f, getattr(ex, 'desc', '')))
                except Exception:
                    print('the profile {} is a concon file'.format(f))
            return
                
        try:
            P = importlib.import_module('profile_scripts.{}'.format(profile))
        except ImportError:
            print('Profile {} not found.'.format(profile))

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
