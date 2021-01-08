import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json, os, glob, sys, time
import bluepyefe as bpefe
import time
import quantities as pq
from neo import AnalogSignal
from neo.io import Spike2IO
from matplotlib.backends.backend_pdf import PdfPages

class single_cell(): 
    """
    Class that binds all functions for trace analysis
    """
    def __init__(self, parent_folder = r'C:\Users\Kamp\Documents\hbp'):
        """
        Set up folder dependencies
        """
        self.log_folder = parent_folder + r'\logs'
        self.raw_folder = parent_folder + r'\data_yuguo'
        self.data_folder = parent_folder + r'\data'
        self.seg_folder = parent_folder + r'\segments'
        self.plot_folder = parent_folder + r'\plots'
        self.meta_folder = parent_folder + r'\metadata'
    
    def convert_data(self, subsampling=4, start=0):
        """
        Converts .smr data in folder to .gz files in outfolder
        :param subsampling, subsamples the data by a factor (default 4)
        :param start, first file in directory that is converted
        """
        folder = self.raw_folder
        outfolder = self.data_folder
        skipped_files = []
        listdir = os.listdir(folder)[start:]
        with open(os.path.join(self.log_folder, 'log_conv.txt'), 'w') as log: 
            for idx, i in enumerate(listdir):
                if not i[-4:] == ".smr":
                    skipped_files.append(i)
                    continue
                f = os.path.join(folder, i)
                img_folder = os.path.join(folder + "_img")
                print("#########")
                print("File " + str(idx) + " out of " + str(len(listdir)))
                print("#########")
                print(f)
                reader = Spike2IO(filename=f)
                bl = reader.read(lazy=False)[0]

                # access to segments
                for idxs, seg in enumerate(bl.segments):
                    d = {}
                    for idxa, asig in enumerate(seg.analogsignals):
                        asig_len = len(asig)
                        time_s = np.array(range(asig_len))
                        flattened = asig.flatten()
                        flt_final = flattened[::subsampling]
                        time_s = time_s[::subsampling]
                        time_s = np.true_divide(time_s, asig.sampling_rate.magnitude)
                        print("signal position in array", idxs, idxa)
                        print("signal units, signals sampling rate => ", asig.units, " - ",
                                asig.sampling_rate)
                        print(flt_final)
                        asig.sampling_rate = asig.sampling_rate * pq.Hz
                        print(time_s)
                        xlabel = "s"
                        ylabel = str(asig.units)[4:]
                        d.update({ylabel:flt_final, 'time': time_s})
                    try:
                        df = pd.DataFrame(d)
                        df.to_csv(outfolder+ '\\' + i[:-4]+'.gz', index=False, compression='infer')
                    except: 
                        print(f"{i[:-4]} could not be converted.")
                        log.write(f'Could not convert {i}')

        print("Skipped files/folder: ", skipped_files)

    def seg_data(self, save_json=False, plot=True):
        folder = self.meta_folder
        metafiles = [file for file in os.listdir(folder) if 'metadata' in file]
        for filename in metafiles:
            # read metadata
            print('Processing: ', filename)
            with open(os.path.join(folder,filename), 'r') as file: 
                meta = json.load(file)
            # load raw data
            data = pd.read_csv(f"{self.data_folder}\\{meta['filename'][:-4]}.gz")
            
            # Only works if metadata unit is millisecond
            fsamp_ms = int(1/(1000*(data.time[1]-data.time[0])))
            nr_segments = len(meta['stimulus_start'])
            pad = 25 # padding in ms
            data.pA -= meta['holding_current'][0]

            traces_dict = {'tonoff':{}, 'traces':{}}
            stim_dict = {}
            for seg in range(nr_segments):
                traces = data.loc[meta['stimulus_start'][seg]*fsamp_ms:meta['stimulus_end'][seg]*fsamp_ms, ['mV','pA']]
                switch = np.diff(traces.pA > meta['stimulus_threshold'][0])!=0
                switch_idx = np.arange(len(traces.pA)-1)[switch]
                for start,end in zip(switch_idx[::2], switch_idx[1::2]): 
                    key = np.round(np.mean(traces.pA.iloc[start:end]),2)
                    stim = traces.pA.iloc[start-pad*fsamp_ms:end+pad*fsamp_ms]
                    trace = traces.mV.iloc[start-pad*fsamp_ms:end+pad*fsamp_ms]
                    traces_dict['tonoff'].update({key:{'ton':[pad],'toff':[pad+end/fsamp_ms]}})
                    traces_dict['traces'].update({key:trace.reset_index(drop=True)})
                    stim_dict.update({key:stim})
            
            traces_dict['traces'] = dict(sorted(traces_dict['traces'].items()))
            traces_dict['tonoff'] = dict(sorted(traces_dict['tonoff'].items()))

            if save_json: 
                with open(os.path.join(self.seg_folder, f'{filename[:-5]}_segments.json'), 'w') as file:
                    json_dict = {}
                    json.dump(traces_dict, file, sort_keys=False)

            df = pd.DataFrame(traces_dict['traces'])
            df.to_csv(os.path.join(self.seg_folder, f'{filename[:-5]}_segments.csv'))            
            
            if plot: 
                nr_traces = 5
                with PdfPages(os.path.join(self.plot_folder, f'{filename[:-5]}_segments.pdf')) as pdf:
                    figs = []
                    for n, key in enumerate(traces_dict['traces'].keys()): 
                        if n%nr_traces == 0: 
                            fig, ax = plt.subplots(figsize=(12,6))
                            figs.append(fig)
                        ax.plot(traces_dict['traces'][key].to_numpy(), label=key, alpha=0.5);
                        ax.legend(loc='upper right')
                        ax.set_title(f"{filename}: {n//5+1}")
                    for fig in figs: 
                        pdf.savefig(fig)
                    plt.close('all')