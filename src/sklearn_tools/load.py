from __future__ import absolute_import
from StringIO import StringIO
import pandas as pd
import os, sys
import numpy.lib
import numpy as np
import pandas as pd
import cPickle as pickle


class BatchLoader(object):
    """
    Loader emits a sequence of pd.DataFrame objects
    """
    def __init__(self, str_or_file, batch_size=100000, **kwargs):
        """
        Construct a loader
        :param str_or_file:
        :param kwargs:
        :return:
        """
        super(BatchLoader, self).__init__()
        assert batch_size > 0, 'Batch size should be greater than 0'
        if isinstance(str_or_file, str):
            str_or_file = open(str_or_file, 'r')
        self.source = str_or_file
        self.batch_size = batch_size
        self.columns = str_or_file.readline().strip().split(kwargs.get('sep', ','))
        self.kwargs = kwargs

    def __iter__(self):
        assert isinstance(self.source, file)
        while True:
            buf = ''.join([self.source.readline() for _ in xrange(self.batch_size)])
            print buf
            df = pd.read_csv(
                StringIO(buf),
                header=None,
                names=self.columns,
                nrows=self.batch_size,
                **self.kwargs
            )
            if len(df):
                yield df
            else:
                return


class MultifileLoader(object):
    """
    Reads all the files in the specified directory and emits them as pd.DataFrame objects
    """

    def __init__(self, path, **kwargs):
        """
        :param path: base dir
        :param kwargs: arguments for pd.read_csv
        :return:
        """
        super(MultifileLoader, self).__init__()
        self.path = path
        self.kwargs = kwargs

    def __iter__(self):
        for root, dirs, files in os.walk(self.path):
            for f in files:
                print >>sys.stderr, 'Reading', f
                full_path = os.path.join(root, f)
                df = pd.read_csv(full_path, **self.kwargs)
                df._metadata = {'path': full_path, 'filename': f}
                df.__tag__ = f
                yield df


def save_pandas(fname, data):
    '''Save DataFrame or Series

    Parameters
    ----------
    fname : str
        filename to use
    data: Pandas DataFrame or Series
    '''
    with open(fname, 'w') as out:
        np.save(out, data)
    if len(data.shape) == 2:
        meta = data.index,data.columns
    elif len(data.shape) == 1:
        meta = (data.index,)
    else:
        raise ValueError('save_pandas: Cannot save this type')
    s = pickle.dumps(meta)
    s = s.encode('string_escape')
    with open(fname, 'a') as f:
        f.seek(0, 2)
        f.write(s)


def load_pandas(fname, mmap_mode='r'):
    '''Load DataFrame or Series

    Parameters
    ----------
    fname : str
        filename
    mmap_mode : str, optional
        Same as numpy.load option
    '''
    values = np.load(fname, mmap_mode=mmap_mode)
    with open(fname) as f:
        numpy.lib.format.read_magic(f)
        numpy.lib.format.read_array_header_1_0(f)
        f.seek(values.dtype.alignment*values.size, 1)
        meta = pickle.loads(f.readline().decode('string_escape'))
    if len(meta) == 2:
        return pd.DataFrame(values, index=meta[0], columns=meta[1])
    elif len(meta) == 1:
        return pd.Series(values, index=meta[0])