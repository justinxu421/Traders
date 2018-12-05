from mymodel import allocate
from dayof.framework.tester import Tester

output_fname = 'output.csv'
test_dir = 'submit_data'

if __name__ == '__main__':
    print('Will load data from directory: ' + test_dir)
    print('Running your allocate function and outputting into: ' + output_fname)
    t = Tester(allocate, test_dir, output_fname)
    t.test()

