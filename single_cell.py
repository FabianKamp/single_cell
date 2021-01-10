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
import shutil as sh
import copy as cp

class single_cell(): 
    """
    Class that binds all functions for trace analysis
    """
    def __init__(self, parent_folder = r'C:\Users\Kamp\Documents\hbp'):
        """
        Set up folder dependencies
        """
        self.log_folder = os.path.join(parent_folder, 'logs')
        self.raw_folder = os.path.join(parent_folder, 'data_yuguo')
        self.data_folder = os.path.join(parent_folder, 'data_csv')
        self.cell_folder = os.path.join(parent_folder, 'data_cells')
        self.seg_folder = os.path.join(parent_folder, 'segments')
        self.plot_folder = os.path.join(parent_folder, 'plots')
        self.meta_folder = os.path.join(parent_folder, 'metadata')

        self.config_temp = os.path.join(parent_folder, 'code', 'config_temp.json')
        self.config_file = os.path.join(parent_folder, 'code', 'config_feature_extraction.json')
    
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

    def setup_folders(self): 
        """
        Function, to copy metafiles and raw data files into common folder.
        """
        metafiles = [file for file in os.listdir(self.meta_folder) if 'metadata' in file]
        with open(os.path.join(self.log_folder, 'log_preparation.txt'), 'w') as log:
            for metafile in metafiles:
                # read metadata
                print('Processing: ', metafile)
                metafile = os.path.join(self.meta_folder, metafile)
                with open(metafile, 'r') as file: 
                    meta = json.load(file)
                
                # copy data and metadata to cell folder
                data_file = os.path.join(self.raw_folder, meta['filename'])
                if not os.path.exists(data_file): 
                    log.write(f'Could not find {data_file}.\n')
                    continue
                foldername = meta['filename'].replace('.smr','')

                folderpath = os.path.join(self.cell_folder, foldername)
                if os.path.exists(folderpath): 
                    print(f'Deleting {folderpath}')
                    sh.rmtree(folderpath)
                os.mkdir(folderpath)
                sh.copy(data_file, folderpath)
                sh.copy(metafile, folderpath)

    def create_config(self): 
        with open(self.config_temp, 'r') as config_temp: 
            config = json.load(config_temp)
        config['path'] = self.cell_folder
        target = config['options']['target']
        cells = [folder for folder in os.listdir(self.cell_folder)] 
        for cell in cells: 
            print('Processing: ', cell)
            folder = os.path.join(self.cell_folder, cell)
            metafile = [file for file in os.listdir(folder) if 'metadata' in file][0]
            try: 
                stims = self.prep_extr(os.path.join(folder, metafile))
                stims = list(np.round(np.array(stims)/1000,3))
                target.extend(stims)
            except:              
                error_message = f"Could not process {cell}"
                print(error_message)
                with open(os.path.join(self.log_folder, 'log_seg.txt'), 'w') as log:
                    log.write(error_message + "\n")
            cell_dict = cp.deepcopy(config['cells']['cell_temp'])
            cell_dict['experiments']['step']['files']=[cell]
            config['cells'][cell]=cell_dict            
        config['options']['target'] = sorted(list(set(target)))
        config['cells'].pop('cell_temp')
        with open(self.config_file, 'w') as config_file: 
            json.dump(config, config_file, indent='\t', sort_keys=False)

    def prep_extr(self, metafile, save_json=True, pad=25):
        """
        This Function reads the metafile, segments original data, saves the segments to 
        a csv/json File and returns the list of applied stimuli.
        
        :params metafile, file containing the metadata
        :params save_json, saves traces and onset and offset times to json file
        :return list of stimulus values in pA
        """
        # read metadata
        with open(metafile, 'r') as file: 
            meta = json.load(file)
        # load raw data
        filename = meta['filename'][:-4]
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
                stim_len = (end-start)/fsamp_ms
                trace = traces.mV.iloc[start-pad*fsamp_ms:end+pad*fsamp_ms]
                traces_dict['tonoff'].update({key:{'ton':[pad],'toff':[pad+stim_len]}})
                traces_dict['traces'].update({key:trace.reset_index(drop=True)})
                stim_dict.update({key:stim})
        
        traces_dict['traces'] = dict(sorted(traces_dict['traces'].items()))
        traces_dict['tonoff'] = dict(sorted(traces_dict['tonoff'].items()))

        if save_json: 
            with open(os.path.join(self.seg_folder, f'{filename}_segments.json'), 'w') as file:
                # convert pd.Series to list, because json can not save pd.Series
                json_dict = traces_dict.copy()
                json_dict['traces'] = {key: list(value) for key, value in json_dict['traces'].items()}
                json.dump(json_dict, file, sort_keys=False)

        df = pd.DataFrame(traces_dict['traces'])
        df.to_csv(os.path.join(self.seg_folder, f'{filename}_segments.csv'))
        
        return list(df.keys())            
    
    def plot_segments(self, nr_traces=5):
        filenames = [file for file in os.listdir(self.seg_folder) if file.endswith('.csv')]   
        for filename in filenames:         
            data = pd.read_csv(os.path.join(self.seg_folder, filename), index_col=0) 
            with PdfPages(os.path.join(self.plot_folder, f'{filename[:-4]}.pdf')) as pdf:
                figs = []
                for n, key in enumerate(data.keys()): 
                    if n%nr_traces == 0: 
                        fig, ax = plt.subplots(figsize=(12,6))
                        figs.append(fig)
                    ax.plot(data[key].to_numpy(), label=key, alpha=0.5);
                    ax.legend(loc='upper right')
                    ax.set_title(f"{filename}: {n//5+1}")
                for fig in figs: 
                    pdf.savefig(fig)
                plt.close('all')
    
    def extr_features(self):
        with open(self.config_file, "r") as file:
            config = json.load(file)

        extractor = bpefe.Extractor('test_run', config)
        extractor.create_dataset()
        extractor.create_metadataset()
        #extractor.plt_traces()
        extractor.extract_features(threshold=-20)
        extractor.mean_features()
        #extractor.analyse_threshold()
        #extractor.plt_features()
        extractor.feature_config_cells(version='legacy')
        extractor.feature_config_all(version='legacy')
        

