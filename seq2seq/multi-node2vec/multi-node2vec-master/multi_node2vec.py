'''
Wrapper for the multi-node2vec algorithm. 

Details can be found in the paper: "Fast Embedding of Multilayer Networks: An Algorithm and Application to Group fMRI" 
by JD Wilson, M Baybay, R Sankar, and P Stillman

Preprint here: https://arxiv.org/pdf/1809.06437.pdf

Contributors:
- Melanie Baybay
University of San Francisco, Department of Computer Science
- Rishi Sankar
Henry M. Gunn High School
- James D. Wilson (maintainer)
University of San Francisco, Department of Mathematics and Statistics

Questions or Bugs? Contact James D. Wilson at jdwilson4@usfca.edu
'''
import os
import src as mltn2v
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-node2vec on multilayer networks.")

    parser.add_argument('--dir', nargs='?', default='data/CONTROL_fmt',
                        help='Absolute path to directory of correlation/adjacency matrix files (csv format). Note that rows and columns must be properly labeled by node ID in each .csv.')

    parser.add_argument('--output', nargs='?', default='results/',
                        help='Absolute path to output directory (no extension).')

    #parser.add_argument('--filename', nargs='?', default='new_results/mltn2v_control',
    #                    help='output filename (no extension).')

    parser.add_argument('--d', type=int, default=50,
                        help='Dimensionality. Default is 100.')

    parser.add_argument('--walk_length', type=int, default=100,
                        help='Length of each random walk. Default is 100.')
                        
    parser.add_argument('--window_size', type=int, default = 10,
                        help='Size of context window used for Skip Gram optimization. Default is 10.')

    parser.add_argument('--n_samples', type=int, default=1,
                        help='Number of walks per node per layer. Default is 1.')

    parser.add_argument('--thresh', type=float, default=0.5,
                        help='Threshold for converting a weighted network to an unweighted one. All weights less than or equal to thresh will be considered 0 and all others 1. Default is 0.5. Use None if the network is unweighted.')

    # parser.add_argument('--w2v_iter', default=1, type=int,
#                         help='Number of epochs in word2vec')

    parser.add_argument('--w2v_workers', type=int, default=8,
                        help='Number of parallel worker threads. Default is 8.')
                        
    parser.add_argument('--rvals', type=float, default=0.4,
                        help='Layer walk parameter for neighborhood search. Default is 0.25')

    parser.add_argument('--pvals', type=float, default=1,
                        help='Return walk parameter for neighborhood search. Default is 1')
    
    parser.add_argument('--qvals', type=float, default=0.5,
                        help='Exploration walk parameter for neighborhood search. Default is 0.50')
  

    return parser.parse_args()


def main(args):
    start = time.time()
    # PARSE LAYERS -- THRESHOLD & CONVERT TO BINARY
    # mltn2v_ Timed in utils and multinode2vec_ The invoke method represents
    # The first parameter is the method description, the second parameter is the method to run, and the return value is the return value of the method
    #So mltn2v. Parse is actually running_ matrix_ Layers method. Layers is actually the data of the edge of a graph
    #layers = mltn2v.timed_invoke("parsing network layers",   # binary=True, thresh=args.thresh
     #                            lambda: mltn2v.parse_matrix_layers(args.dir, binary=False, thresh=0))
    # check if layers were parsed
    layers=True
    if layers:
        # EXTRACT NEIGHBORHOODS
        nbrhd_dict = mltn2v.timed_invoke("extracting neighborhoods",
                                     lambda: mltn2v.extract_neighborhoods_walk(layers, args.walk_length, args.rvals, args.pvals, args.qvals))
        # GENERATE FEATURES
        out = mltn2v.clean_output(args.output)
        for w in args.rvals:
            out_path = os.path.join(out, 'r' + str(w) + '/mltn2v_results') 
            mltn2v.timed_invoke("generating features",
                                lambda: mltn2v.generate_features(nbrhd_dict[w], args.d, out_path, nbrhd_size=args.window_size,
                                                                 w2v_iter=1, workers=args.w2v_workers))

            print("\nCompleted Multilayer Network Embedding for r=" + str(w) + " in {:.2f} secs.\nSee results:".format(time.time() - start))
            print("\t" + out_path + ".csv")
        print("Completed Multilayer Network Embedding for all r values.")
    else:
        print("Whoops!")


if __name__ == '__main__':
    args = parse_args()
    args.rvals = [args.rvals]
    main(args)
