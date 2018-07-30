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
    save_folder + "dist_only_eval_@2018-07-30_09:46:01.922_full.json"
    # save_folder + "dist_only_eval_@2018-07-23_11:57:09.734_full.json"

    all_scores = load_json_from_file( filepath)

    train_plot = False
    eval_plot = True
    #If find all the run's respective scores
    if ("train_scores" in all_scores.keys()) and train_plot:
        print( "### DEBUG: Training scores found")

        all_training_scores = all_scores["train_scores"]

        print( "###DEBUG: Runs count in training scores: %d" % len( all_training_scores))

        f, a = plt.subplots( len( all_training_scores), sharex=True)

        # GEneral legend
        for trun_indx, trun_scores in enumerate( all_training_scores):
            a[trun_indx].set_title( "Run {} training scores; Total episodes: {}"
                .format( trun_indx, len( trun_scores)))

            a[trun_indx].plot( [ i for i in range( len( trun_scores))],
                trun_scores)

    # if finds run's respective training scores
    if ("eval_scores" in all_scores.keys()) and eval_plot:
        print( "### DEBUG: Evaluation scores found")

        all_eval_scores = all_scores["eval_scores"]

        f2 = plt.figure(2)

        plt.plot( [ i for i in range( len( all_eval_scores))],
            [ np.mean( np.sort( evrun_scores)[::-1][0:3])
                for evrun_scores in all_eval_scores])

        plt.title( "Avg Top 3 Score out of 10 - Torcs - Rwrd = Dist. Only")
        plt.xlabel( "Run")
        plt.ylabel( "Score")

        # for evrun, evrun_scores in enumerate( all_eval_scores):
        #     print( "Evalutation Run %d - Avg Top 3 Score: %.2f" %
        #         ( evrun, np.mean( np.sort( evrun_scores)[::-1][0:3])))
        #
        #     plt.figure(evrun)
        #     plt.plot( [i for i in range( len( evrun_scores))], evrun_scores)
        #     plt.show()

    plt.show()
