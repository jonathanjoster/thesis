import numpy as np
from StylizeModel import StylizeModel
from keras.utils.vis_utils import plot_model
import argparse

def main(plot, evaluate, rmse, rec_row, compare_extent):
    print(args)
    eegnhyp = np.load('./data/eegnhyp.npz')
    eeg, hyp = eegnhyp['arr_0'], eegnhyp['arr_1']
    xny = np.load('./data_sec/test_data.npz')
    x, y = xny['arr_0'], xny['arr_1']
    
    model = StylizeModel(eeg, hyp, x, y)
    if plot:
        plot_model(model.model, show_shapes=True, show_layer_names=False)
    if evaluate:
        model.evaluate()
    if rmse:
        model.show_rmse_transition()
    if rec_row != 0:
        model.show_rec_transition(row=rec_row)
    if compare_extent!= 0:
        if compare_extent > 32 or compare_extent < 0:
            raise ValueError('0 <= n <= 32')
        model.compare(idx=212, extent=compare_extent)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dnn model for signal style transfer')
    parser.add_argument('--plot', action='store_true', help='plot model')
    parser.add_argument('--evaluate', action='store_true', help='evaluate model')
    parser.add_argument('--rmse', action='store_true', help='show rmse transition')
    parser.add_argument('--rec', type=int, default=0, help='show n row(s) of rec transition')
    parser.add_argument('--compare', type=int, default=0, help='comparison between raw and n-level stylized rec signal')
    args = parser.parse_args()
    main(args.plot, args.evaluate, args.rmse, args.rec, args.compare)