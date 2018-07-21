import json
import numpy as np
import matplotlib.pyplot as plt
import os

save_folder = "training_data/"

#Misc tools
def load_json_from_file( filepath):
    json_data = None
    try:
        with open( filepath, "r") as f:
            json_data = json.load( f)
    except Exception as ex:
        print( ex)

    return json_data

#MAIN
if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__)) + "/" + \
        save_folder + "dist_only_eval_@2018-07-21_17:08:48.543_full.json"

    all_eval_scores = load_json_from_file( filepath)["eval_scores"]

    # print( all_eval_scores)

    plt.plot( [ i for i in range( len( all_eval_scores))],
        [ np.mean( np.sort( evrun_scores)[::-1][0:3])
            for evrun_scores in all_eval_scores])

    plt.title( "Avg Top 3 Score out of 10 - Torcs - Rwrd = Dist. Only")
    plt.xlabel( "Run")
    plt.ylabel( "Score")
    plt.show()
    # for evrun, evrun_scores in enumerate( all_eval_scores):
    #     print( "Evalutation Run %d - Avg Top 3 Score: %.2f" %
    #         ( evrun, np.mean( np.sort( evrun_scores)[::-1][0:3])))
    #
    #     plt.figure(evrun)
    #     plt.plot( [i for i in range( len( evrun_scores))], evrun_scores)
    #     plt.show()
