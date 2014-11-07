import cPickle
import os, sys


class Workspace(object):
    def __init__(self, base_dir, globals):
        super(Workspace, self).__init__()
        assert os.path.exists(base_dir)
        assert os.path.isdir(base_dir)
        self.base = base_dir
        self.vars = set()
        self.versions = {}
        self.globals = globals

    def register(self, *args):
        for arg in args:
            self.vars.add(arg)

    def checkpoint(self):
        for var in self.vars:
            try:
                if var not in self.globals():
                    print >>sys.stderr, 'Unknown variable', var
                    continue
                if id(self.globals()[var]) == self.versions.get(var, None):
                    print >>sys.stderr, 'No changes detected for', var
                    continue
                with open(os.path.join(self.base, var), 'w') as out:
                    print >>sys.stderr, 'Saving', var
                    cPickle.dump(self.globals()[var], out, protocol=2)
            except BaseException as e:
                print >>sys.stderr, 'Unable to save', var
                print >>sys.stderr, repr(e)

    def restore(self, *vars):
        for var in vars if vars else self.vars:
            path = os.path.join(self.base, var)
            if not os.path.exists(path):
                print >>sys.stderr, 'No data found for', var
                continue
            with open(path, 'r') as input:
                self.globals()[var] = cPickle.load(input)
                self.versions[var] = id(self.globals()[var])
                print >>sys.stderr, 'Restored', var

    def is_known(self, var):
        return os.path.exists(os.path.join(self.base, var))

    def unregister(self, *args):
        for arg in args:
            self.vars.remove(arg)

    def clear(self, *vars):
        for var in vars if vars else self.vars:
            os.remove(os.path.join(self.base, var))
